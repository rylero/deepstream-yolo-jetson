/*
 * ds_nt_probe.c
 *
 * DeepStream YOLOv11n pipeline with NetworkTables publisher and MJPEG HTTP stream.
 *
 * Source mode is selected by the VIDEO_URI environment variable:
 *   VIDEO_URI set   → nvurisrcbin (file://, rtsp://, http://)
 *   VIDEO_URI unset → v4l2src (/dev/video0, USB camera)
 *
 * Pipeline (file mode):
 *   nvurisrcbin ──(pad-added)──> nvstreammux → nvinfer → tee ─┬→ queue → fakesink
 *                                              ^─ probe→NT      └→ queue → nvvideoconvert
 *                                                                          → videoconvert
 *                                                                          → jpegenc → appsink → HTTP :8080
 *
 * Pipeline (camera mode):
 *   v4l2src → capsfilter → nvvideoconvert → nvstreammux → nvinfer → tee ─┬→ ...
 *                                                         ^─ probe→NT      └→ ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <glib.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include <ntcore.h>

#define WPI_STR(s) ((struct WPI_String){ (s), sizeof(s) - 1 })

#include "detection_types.h"

#define INFER_CONFIG  "/opt/deepstream/config/config_infer_primary_yolo11n.txt"
#define VIDEO_DEVICE  "/dev/video0"
#define SAMPLE_URI    "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4"

/* ------------------------------------------------------------------ */
/* Globals                                                              */
/* ------------------------------------------------------------------ */

static GMainLoop   *g_loop     = NULL;
static NT_Inst      g_nt_inst  = 0;
static NT_Publisher g_nt_pub   = 0;
static guint32      g_frame_no = 0;
static float        g_fps      = 0.0f;
static int64_t      g_last_ts  = 0;

#define FPS_ALPHA 0.1f

static int64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}

/* ------------------------------------------------------------------ */
/* MJPEG HTTP server                                                    */
/* ------------------------------------------------------------------ */

#define MJPEG_PORT     8080
#define MJPEG_BOUNDARY "mjpegboundary"

typedef struct { guint8 *data; gsize size; } JpegFrame;

static JpegFrame       g_jpeg       = { NULL, 0 };
static pthread_mutex_t g_jpeg_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_jpeg_cond  = PTHREAD_COND_INITIALIZER;
static gboolean        g_jpeg_ready = FALSE;

static GstFlowReturn on_new_jpeg_sample(GstElement *sink, gpointer user_data)
{
    (void)user_data;
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        pthread_mutex_lock(&g_jpeg_mutex);
        g_free(g_jpeg.data);
        g_jpeg.data  = (guint8 *)g_memdup2(map.data, map.size);
        g_jpeg.size  = map.size;
        g_jpeg_ready = TRUE;
        pthread_cond_broadcast(&g_jpeg_cond);
        pthread_mutex_unlock(&g_jpeg_mutex);
        gst_buffer_unmap(buf, &map);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static const char *HTML_PAGE =
    "<!DOCTYPE html><html><head><title>DeepStream Vision</title>"
    "<style>body{margin:0;background:#111;display:flex;flex-direction:column;"
    "align-items:center;justify-content:center;min-height:100vh;color:#eee;"
    "font-family:sans-serif;gap:12px;}img{max-width:100%;border:2px solid #444;"
    "border-radius:4px;}h2{margin:0;font-size:1.2rem;}</style></head>"
    "<body><h2>DeepStream YOLOv11n Live Feed</h2>"
    "<img src=\"/stream\" alt=\"Live stream\"></body></html>";

static void send_all(int fd, const void *data, size_t len)
{
    const char *p = (const char *)data;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return;
        p += n; len -= (size_t)n;
    }
}

static void *handle_client(void *arg)
{
    int fd = *(int *)arg;
    g_free(arg);

    char req[512] = {0};
    recv(fd, req, sizeof(req) - 1, 0);

    if (strstr(req, "GET / ") || strstr(req, "GET /\r")) {
        char hdr[256];
        int hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            "Content-Length: %zu\r\nConnection: close\r\n\r\n", strlen(HTML_PAGE));
        send_all(fd, hdr, (size_t)hlen);
        send_all(fd, HTML_PAGE, strlen(HTML_PAGE));

    } else if (strstr(req, "GET /stream")) {
        const char *hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace;boundary=" MJPEG_BOUNDARY "\r\n"
            "Cache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
        send_all(fd, hdr, strlen(hdr));

        while (1) {
            pthread_mutex_lock(&g_jpeg_mutex);
            while (!g_jpeg_ready)
                pthread_cond_wait(&g_jpeg_cond, &g_jpeg_mutex);
            guint8 *frame = (guint8 *)g_memdup2(g_jpeg.data, g_jpeg.size);
            gsize   fsz   = g_jpeg.size;
            g_jpeg_ready  = FALSE;
            pthread_mutex_unlock(&g_jpeg_mutex);

            char part[256];
            int plen = snprintf(part, sizeof(part),
                "--" MJPEG_BOUNDARY "\r\nContent-Type: image/jpeg\r\n"
                "Content-Length: %zu\r\n\r\n", fsz);
            if (send(fd, part,  (size_t)plen, MSG_NOSIGNAL) <= 0 ||
                send(fd, frame, fsz,           MSG_NOSIGNAL) <= 0 ||
                send(fd, "\r\n", 2,            MSG_NOSIGNAL) <= 0) {
                g_free(frame); break;
            }
            g_free(frame);
        }
    } else {
        send_all(fd,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            72);
    }
    close(fd);
    return NULL;
}

static void *http_server_thread(void *arg)
{
    (void)arg;
    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("[HTTP] socket"); return NULL; }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(MJPEG_PORT);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[HTTP] bind"); close(srv); return NULL;
    }
    listen(srv, 10);
    printf("[HTTP] MJPEG stream →  http://<jetson-ip>:%d/\n", MJPEG_PORT);

    while (1) {
        int client = accept(srv, NULL, NULL);
        if (client < 0) continue;
        int *pfd = (int *)g_malloc(sizeof(int));
        *pfd = client;
        pthread_t t;
        pthread_create(&t, NULL, handle_client, pfd);
        pthread_detach(t);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* NetworkTables                                                         */
/* ------------------------------------------------------------------ */

static void nt_init(void)
{
    const char *team_str = getenv("NT_TEAM_NUMBER");
    unsigned int team    = team_str ? (unsigned int)atoi(team_str) : 0;

    struct WPI_String identity   = { "jetson-vision",       13 };
    struct WPI_String topic_name = { "/vision/detections",  18 };
    struct WPI_String type_str   = { "raw",                  3 };
    struct WPI_String fallback   = { "roborio-0-frc.local", 19 };

    g_nt_inst = NT_GetDefaultInstance();
    NT_StartClient4(g_nt_inst, &identity);

    if (team > 0) {
        printf("[NT] Connecting to team %u roboRIO\n", team);
        NT_SetServerTeam(g_nt_inst, team, 0);
    } else {
        fprintf(stderr, "[NT] Warning: NT_TEAM_NUMBER not set, trying roborio-0-frc.local\n");
        NT_SetServer(g_nt_inst, &fallback, 1735);
    }

    NT_Topic topic = NT_GetTopic(g_nt_inst, &topic_name);
    struct NT_PubSubOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.structSize = sizeof(opts);
    g_nt_pub = NT_Publish(topic, NT_RAW, &type_str, &opts);
    printf("[NT] Publisher created for /vision/detections\n");
}

/* ------------------------------------------------------------------ */
/* Pad probe (nvinfer src)                                              */
/* ------------------------------------------------------------------ */

static GstPadProbeReturn inference_src_pad_buffer_probe(
    GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    (void)pad; (void)user_data;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    int64_t now = now_us();
    if (g_last_ts > 0) {
        float inst = 1000000.0f / (float)(now - g_last_ts);
        g_fps = (g_fps == 0.0f) ? inst : FPS_ALPHA * inst + (1.0f - FPS_ALPHA) * g_fps;
    }
    g_last_ts = now;

    NvDsFrameMetaList *fl = batch_meta->frame_meta_list;
    while (fl) {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)fl->data;
        if (!fm) { fl = fl->next; continue; }

        DetectionFrame det = {0};
        det.frame_number = g_frame_no++;
        det.timestamp_us = now;
        det.fps          = g_fps;

        float fw = fm->source_frame_width  > 0 ? (float)fm->source_frame_width  : (float)FRAME_WIDTH;
        float fh = fm->source_frame_height > 0 ? (float)fm->source_frame_height : (float)FRAME_HEIGHT;

        NvDsObjectMetaList *ol = fm->obj_meta_list;
        while (ol) {
            NvDsObjectMeta *om = (NvDsObjectMeta *)ol->data;
            if (om && det.num_detections < MAX_DETECTIONS) {
                Detection *d = &det.detections[det.num_detections++];
                d->class_id   = om->class_id;
                d->confidence = om->confidence;
                d->left       = om->rect_params.left   / fw;
                d->top        = om->rect_params.top    / fh;
                d->width      = om->rect_params.width  / fw;
                d->height     = om->rect_params.height / fh;
                strncpy(d->label, om->obj_label, sizeof(d->label) - 1);
            }
            ol = ol->next;
        }

        NT_SetRaw(g_nt_pub, 0, (const uint8_t *)&det, sizeof(det));

        if (det.num_detections > 0)
            printf("[DS] Frame %u: %u detection(s)  %.1f fps\n",
                   det.frame_number, det.num_detections, g_fps);

        fl = fl->next;
    }
    return GST_PAD_PROBE_OK;
}

/* ------------------------------------------------------------------ */
/* Dynamic pad callback (nvurisrcbin → nvstreammux)                     */
/* ------------------------------------------------------------------ */

static void on_pad_added(GstElement *src, GstPad *new_pad, gpointer data)
{
    (void)src;
    GstElement *streammux = (GstElement *)data;

    /* Only link video pads */
    GstCaps *caps = gst_pad_get_current_caps(new_pad);
    if (!caps) caps = gst_pad_query_caps(new_pad, NULL);
    const gchar *mime = gst_structure_get_name(gst_caps_get_structure(caps, 0));
    gboolean is_video = g_str_has_prefix(mime, "video/");
    gst_caps_unref(caps);
    if (!is_video) return;

    GstPad *sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    if (!sinkpad) {
        fprintf(stderr, "[Error] Could not get streammux sink_0\n");
        return;
    }
    if (gst_pad_link(new_pad, sinkpad) != GST_PAD_LINK_OK)
        fprintf(stderr, "[Error] Failed to link uri-source → streammux\n");

    gst_object_unref(sinkpad);
}

/* ------------------------------------------------------------------ */
/* Bus handler                                                           */
/* ------------------------------------------------------------------ */

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    (void)bus;
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
        printf("[GStreamer] End-of-stream\n");
        g_main_loop_quit(loop);
        break;
    case GST_MESSAGE_ERROR: {
        gchar *dbg = NULL; GError *err = NULL;
        gst_message_parse_error(msg, &err, &dbg);
        fprintf(stderr, "[GStreamer] Error: %s\n", err->message);
        if (dbg) fprintf(stderr, "[GStreamer] Debug: %s\n", dbg);
        g_error_free(err); g_free(dbg);
        g_main_loop_quit(loop);
        break;
    }
    default: break;
    }
    return TRUE;
}

static void sigint_handler(int sig) { (void)sig; if (g_loop) g_main_loop_quit(g_loop); }

/* ------------------------------------------------------------------ */
/* Main                                                                  */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);
    g_loop = g_main_loop_new(NULL, FALSE);
    nt_init();

    const char *video_uri = getenv("VIDEO_URI");
    if (!video_uri) video_uri = "";
    gboolean use_file = (video_uri[0] != '\0');

    if (use_file)
        printf("[DS] Source: %s\n", video_uri);
    else
        printf("[DS] Source: v4l2 camera (%s)\n", VIDEO_DEVICE);

    /* ---- Create elements ---- */
    GstElement *pipeline  = gst_pipeline_new("ds-nt-pipeline");
    GstElement *source    = NULL;
    GstElement *capsfilter= NULL;
    GstElement *vidconv   = NULL;   /* only for camera mode */
    GstElement *streammux = gst_element_factory_make("nvstreammux",    "stream-muxer");
    GstElement *infer     = gst_element_factory_make("nvinfer",        "primary-nvinference");
    GstElement *tee       = gst_element_factory_make("tee",            "post-infer-tee");
    GstElement *queue_inf = gst_element_factory_make("queue",          "queue-infer");
    GstElement *sink      = gst_element_factory_make("fakesink",       "fake-sink");
    GstElement *queue_web = gst_element_factory_make("queue",          "queue-web");
    GstElement *nv2cpu    = gst_element_factory_make("nvvideoconvert", "nv-to-cpu");
    GstElement *vcpucpu   = gst_element_factory_make("videoconvert",   "cpu-conv");
    GstElement *jpegenc   = gst_element_factory_make("jpegenc",        "jpeg-enc");
    GstElement *appsink   = gst_element_factory_make("appsink",        "web-appsink");

    if (use_file) {
        source = gst_element_factory_make("nvurisrcbin", "uri-source");
        if (!source) {
            fprintf(stderr, "[Error] Could not create nvurisrcbin\n");
            return -1;
        }
        g_object_set(source, "uri", video_uri, "gpu-id", 0, NULL);
        g_signal_connect(source, "pad-added", G_CALLBACK(on_pad_added), streammux);
    } else {
        source     = gst_element_factory_make("v4l2src",        "v4l2-source");
        capsfilter = gst_element_factory_make("capsfilter",     "caps-filter");
        vidconv    = gst_element_factory_make("nvvideoconvert", "nv-vidconv");
        if (!source || !capsfilter || !vidconv) {
            fprintf(stderr, "[Error] Could not create camera elements\n");
            return -1;
        }
        g_object_set(source, "device", VIDEO_DEVICE, NULL);

        GstCaps *caps = gst_caps_from_string(
            "video/x-raw,width=" G_STRINGIFY(FRAME_WIDTH)
            ",height=" G_STRINGIFY(FRAME_HEIGHT)
            ",framerate=30/1,format=YUY2");
        g_object_set(capsfilter, "caps", caps, NULL);
        gst_caps_unref(caps);
    }

    if (!pipeline || !streammux || !infer || !tee ||
        !queue_inf || !sink || !queue_web || !nv2cpu || !vcpucpu || !jpegenc || !appsink) {
        fprintf(stderr, "[Error] Failed to create pipeline elements\n");
        return -1;
    }

    /* ---- Configure ---- */
    g_object_set(streammux,
                 "batch-size",           1,
                 "width",                FRAME_WIDTH,
                 "height",               FRAME_HEIGHT,
                 "batched-push-timeout", 40000,
                 "live-source",          use_file ? FALSE : TRUE,
                 NULL);
    g_object_set(infer,   "config-file-path", INFER_CONFIG, NULL);
    g_object_set(sink,    "sync",             FALSE,        NULL);
    g_object_set(jpegenc, "quality",          50,           NULL);
    g_object_set(appsink,
                 "emit-signals", TRUE, "sync", FALSE,
                 "max-buffers",  1,    "drop", TRUE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_jpeg_sample), NULL);

    /* ---- Add elements to bin ---- */
    if (use_file) {
        gst_bin_add_many(GST_BIN(pipeline),
                         source, streammux, infer, tee,
                         queue_inf, sink,
                         queue_web, nv2cpu, vcpucpu, jpegenc, appsink,
                         NULL);
    } else {
        gst_bin_add_many(GST_BIN(pipeline),
                         source, capsfilter, vidconv, streammux, infer, tee,
                         queue_inf, sink,
                         queue_web, nv2cpu, vcpucpu, jpegenc, appsink,
                         NULL);
    }

    /* ---- Link inference path ---- */
    if (!use_file) {
        /* camera: v4l2src → capsfilter → nvvideoconvert → streammux (request pad) */
        if (!gst_element_link_many(source, capsfilter, vidconv, NULL)) {
            fprintf(stderr, "[Error] Failed to link camera source chain\n");
            return -1;
        }
        GstPad *mux_sink = gst_element_request_pad_simple(streammux, "sink_0");
        GstPad *vc_src   = gst_element_get_static_pad(vidconv, "src");
        if (gst_pad_link(vc_src, mux_sink) != GST_PAD_LINK_OK) {
            fprintf(stderr, "[Error] Failed to link vidconv → streammux\n");
            return -1;
        }
        gst_object_unref(vc_src);
        gst_object_unref(mux_sink);
    }
    /* file: nvurisrcbin → streammux linked via on_pad_added callback */

    /* streammux → nvinfer → tee */
    if (!gst_element_link_many(streammux, infer, tee, NULL)) {
        fprintf(stderr, "[Error] Failed to link streammux → infer → tee\n");
        return -1;
    }

    /* tee → queue_inf → fakesink */
    if (!gst_element_link_many(tee, queue_inf, sink, NULL)) {
        fprintf(stderr, "[Error] Failed to link tee → queue_inf → fakesink\n");
        return -1;
    }

    /* tee → queue_web → nvvideoconvert → videoconvert → jpegenc → appsink */
    if (!gst_element_link(tee, queue_web)) {
        fprintf(stderr, "[Error] Failed to link tee → queue_web\n");
        return -1;
    }

    /* nvvideoconvert must output CPU memory for videoconvert — use caps filter */
    GstCaps *cpu_caps = gst_caps_from_string("video/x-raw,format=I420");
    if (!gst_element_link_filtered(nv2cpu, vcpucpu, cpu_caps)) {
        fprintf(stderr, "[Error] Failed to link nv2cpu → vcpucpu\n");
        return -1;
    }
    gst_caps_unref(cpu_caps);

    if (!gst_element_link(queue_web, nv2cpu) ||
        !gst_element_link_many(vcpucpu, jpegenc, appsink, NULL)) {
        fprintf(stderr, "[Error] Failed to link web branch\n");
        return -1;
    }

    /* ---- Pad probe on nvinfer src ---- */
    GstPad *infer_src = gst_element_get_static_pad(infer, "src");
    if (!infer_src) { fprintf(stderr, "[Error] No nvinfer src pad\n"); return -1; }
    gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER,
                      inference_src_pad_buffer_probe, NULL, NULL);
    gst_object_unref(infer_src);

    /* ---- Bus + signals ---- */
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, g_loop);
    gst_object_unref(bus);

    signal(SIGINT,  sigint_handler);
    signal(SIGTERM, sigint_handler);

    /* ---- Start HTTP server ---- */
    pthread_t http_tid;
    pthread_create(&http_tid, NULL, http_server_thread, NULL);
    pthread_detach(http_tid);

    /* ---- Play ---- */
    printf("[DS] Starting pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(g_loop);

    /* ---- Cleanup ---- */
    printf("[DS] Shutting down...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    g_main_loop_unref(g_loop);
    NT_StopClient(g_nt_inst);
    return 0;
}
