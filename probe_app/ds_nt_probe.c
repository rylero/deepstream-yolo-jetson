/*
 * ds_nt_probe.c
 *
 * DeepStream YOLOv11n inference pipeline with NetworkTables publisher
 * and built-in MJPEG HTTP stream for driver-station viewing.
 *
 * Pipeline:
 *   v4l2src → capsfilter → tee ─┬→ queue → nvvideoconvert → nvstreammux
 *                                │              → nvinfer → fakesink
 *                                │         ^── pad probe publishes NT_RAW
 *                                └→ queue → videoconvert → jpegenc → appsink
 *                                                          ^── HTTP thread serves
 *                                                              MJPEG on port 8080
 *
 * Build: see CMakeLists.txt
 * Run:   NT_TEAM_NUMBER=XXXX ./ds_nt_probe
 * View:  http://<jetson-ip>:8080/
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

/* DeepStream metadata headers (from /opt/nvidia/deepstream/deepstream/sources/includes) */
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

/* WPILib NetworkTables C API */
#include <ntcore.h>

/* WPILib 2025 uses WPI_String instead of const char* for string args */
#define WPI_STR(s) ((struct WPI_String){ (s), sizeof(s) - 1 })

#include "detection_types.h"

/* ------------------------------------------------------------------ */
/* Globals                                                              */
/* ------------------------------------------------------------------ */

static GMainLoop  *g_loop     = NULL;
static NT_Inst     g_nt_inst  = 0;
static NT_Publisher g_nt_pub  = 0;
static guint32     g_frame_no = 0;
static float       g_fps      = 0.0f;
static int64_t     g_last_ts  = 0;   /* microseconds, for FPS calculation */

#define FPS_ALPHA 0.1f   /* EMA smoothing factor */

static int64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}

#define INFER_CONFIG "/opt/deepstream/config/config_infer_primary_yolo11n.txt"
#define VIDEO_DEVICE "/dev/video0"

/* ------------------------------------------------------------------ */
/* MJPEG HTTP server                                                    */
/* ------------------------------------------------------------------ */

#define MJPEG_PORT     8080
#define MJPEG_BOUNDARY "mjpegboundary"

typedef struct {
    guint8 *data;
    gsize   size;
} JpegFrame;

static JpegFrame       g_jpeg        = { NULL, 0 };
static pthread_mutex_t g_jpeg_mutex  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_jpeg_cond   = PTHREAD_COND_INITIALIZER;
static gboolean        g_jpeg_ready  = FALSE;

/* Called by appsink on every new JPEG buffer */
static GstFlowReturn on_new_jpeg_sample(GstElement *sink, gpointer user_data)
{
    (void)user_data;

    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample)
        return GST_FLOW_OK;

    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    pthread_mutex_lock(&g_jpeg_mutex);
    g_free(g_jpeg.data);
    g_jpeg.data  = (guint8 *)g_memdup2(map.data, map.size);
    g_jpeg.size  = map.size;
    g_jpeg_ready = TRUE;
    pthread_cond_broadcast(&g_jpeg_cond);
    pthread_mutex_unlock(&g_jpeg_mutex);

    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static const char *HTML_PAGE =
    "<!DOCTYPE html><html><head><title>DeepStream Vision</title>"
    "<style>"
    "body{margin:0;background:#111;display:flex;flex-direction:column;"
    "align-items:center;justify-content:center;min-height:100vh;"
    "color:#eee;font-family:sans-serif;gap:12px;}"
    "img{max-width:100%;border:2px solid #444;border-radius:4px;}"
    "h2{margin:0;font-size:1.2rem;letter-spacing:0.05em;}"
    "</style></head>"
    "<body>"
    "<h2>DeepStream YOLOv11n Live Feed</h2>"
    "<img src=\"/stream\" alt=\"Live stream\">"
    "</body></html>";

static void send_all(int fd, const void *data, size_t len)
{
    const char *p = (const char *)data;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n <= 0) return;
        p   += n;
        len -= (size_t)n;
    }
}

static void *handle_client(void *arg)
{
    int fd = *(int *)arg;
    g_free(arg);

    /* Read HTTP request */
    char req[512] = {0};
    recv(fd, req, sizeof(req) - 1, 0);

    gboolean is_stream = (strstr(req, "GET /stream") != NULL);
    gboolean is_root   = (strstr(req, "GET / ")      != NULL ||
                          strstr(req, "GET /\r")      != NULL);

    if (is_root) {
        char hdr[256];
        int  hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n\r\n",
            strlen(HTML_PAGE));
        send_all(fd, hdr,       (size_t)hlen);
        send_all(fd, HTML_PAGE, strlen(HTML_PAGE));

    } else if (is_stream) {
        const char *resp_hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace;boundary=" MJPEG_BOUNDARY "\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n\r\n";
        send_all(fd, resp_hdr, strlen(resp_hdr));

        while (1) {
            pthread_mutex_lock(&g_jpeg_mutex);
            while (!g_jpeg_ready)
                pthread_cond_wait(&g_jpeg_cond, &g_jpeg_mutex);
            guint8 *frame = (guint8 *)g_memdup2(g_jpeg.data, g_jpeg.size);
            gsize   fsz   = g_jpeg.size;
            g_jpeg_ready  = FALSE;
            pthread_mutex_unlock(&g_jpeg_mutex);

            char part_hdr[256];
            int  phlen = snprintf(part_hdr, sizeof(part_hdr),
                "--" MJPEG_BOUNDARY "\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: %zu\r\n\r\n", fsz);

            if (send(fd, part_hdr, (size_t)phlen, MSG_NOSIGNAL) <= 0 ||
                send(fd, frame,    fsz,            MSG_NOSIGNAL) <= 0 ||
                send(fd, "\r\n",   2,              MSG_NOSIGNAL) <= 0) {
                g_free(frame);
                break;
            }
            g_free(frame);
        }
    } else {
        const char *not_found =
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
            "Connection: close\r\n\r\n";
        send_all(fd, not_found, strlen(not_found));
    }

    close(fd);
    return NULL;
}

static void *http_server_thread(void *arg)
{
    (void)arg;

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("[HTTP] socket"); return NULL; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(MJPEG_PORT);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[HTTP] bind"); close(server_fd); return NULL;
    }
    listen(server_fd, 10);

    printf("[HTTP] MJPEG stream →  http://<jetson-ip>:%d/\n", MJPEG_PORT);

    while (1) {
        int client_fd = accept(server_fd, NULL, NULL);
        if (client_fd < 0) continue;

        int *pfd = (int *)g_malloc(sizeof(int));
        *pfd = client_fd;
        pthread_t t;
        pthread_create(&t, NULL, handle_client, pfd);
        pthread_detach(t);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* NetworkTables setup                                                  */
/* ------------------------------------------------------------------ */

static void nt_init(void)
{
    const char *team_str = getenv("NT_TEAM_NUMBER");
    unsigned int team    = team_str ? (unsigned int)atoi(team_str) : 0;

    /* Use local variables for WPI_String — avoids compound-literal lifetime issues */
    struct WPI_String identity   = { "jetson-vision",       13 };
    struct WPI_String topic_name = { "/vision/detections",  18 };
    struct WPI_String type_str   = { "raw",                  3 };
    struct WPI_String fallback   = { "roborio-0-frc.local", 19 };

    g_nt_inst = NT_GetDefaultInstance();
    NT_StartClient4(g_nt_inst, &identity);

    if (team > 0) {
        printf("[NT] Connecting to team %u roboRIO\n", team);
        NT_SetServerTeam(g_nt_inst, team, 0);  /* 0 = default port 1735 */
    } else {
        fprintf(stderr, "[NT] Warning: NT_TEAM_NUMBER not set. Trying roborio-0-frc.local\n");
        NT_SetServer(g_nt_inst, &fallback, 1735);
    }

    NT_Topic topic = NT_GetTopic(g_nt_inst, &topic_name);

    /* NT_Publish requires a valid (non-NULL) options pointer in WPILib 2025 */
    struct NT_PubSubOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.structSize = sizeof(opts);
    g_nt_pub = NT_Publish(topic, NT_RAW, &type_str, &opts);

    printf("[NT] Publisher created for /vision/detections\n");
}

/* ------------------------------------------------------------------ */
/* DeepStream pad probe                                                 */
/* ------------------------------------------------------------------ */

static GstPadProbeReturn inference_src_pad_buffer_probe(
    GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    (void)pad;
    (void)user_data;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf)
        return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta)
        return GST_PAD_PROBE_OK;

    /* Compute timestamp and FPS once per batch */
    int64_t now = now_us();
    if (g_last_ts > 0) {
        float instant_fps = 1000000.0f / (float)(now - g_last_ts);
        g_fps = (g_fps == 0.0f) ? instant_fps
                                 : FPS_ALPHA * instant_fps + (1.0f - FPS_ALPHA) * g_fps;
    }
    g_last_ts = now;

    NvDsFrameMetaList *frame_list = batch_meta->frame_meta_list;
    while (frame_list) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)frame_list->data;
        if (!frame_meta) {
            frame_list = frame_list->next;
            continue;
        }

        DetectionFrame det_frame;
        memset(&det_frame, 0, sizeof(det_frame));
        det_frame.frame_number   = g_frame_no++;
        det_frame.num_detections = 0;
        det_frame.timestamp_us   = now;
        det_frame.fps            = g_fps;

        float frame_w = (float)frame_meta->source_frame_width;
        float frame_h = (float)frame_meta->source_frame_height;
        /* Guard against zero dimensions (shouldn't happen, but be safe) */
        if (frame_w <= 0.0f) frame_w = (float)FRAME_WIDTH;
        if (frame_h <= 0.0f) frame_h = (float)FRAME_HEIGHT;

        NvDsObjectMetaList *obj_list = frame_meta->obj_meta_list;
        while (obj_list) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)obj_list->data;
            if (obj_meta && det_frame.num_detections < MAX_DETECTIONS) {
                Detection *d = &det_frame.detections[det_frame.num_detections];

                d->class_id   = obj_meta->class_id;
                d->confidence = obj_meta->confidence;
                d->left       = obj_meta->rect_params.left   / frame_w;
                d->top        = obj_meta->rect_params.top    / frame_h;
                d->width      = obj_meta->rect_params.width  / frame_w;
                d->height     = obj_meta->rect_params.height / frame_h;

                /* Copy label, ensure null-terminated */
                strncpy(d->label, obj_meta->obj_label, sizeof(d->label) - 1);
                d->label[sizeof(d->label) - 1] = '\0';

                det_frame.num_detections++;
            }
            obj_list = obj_list->next;
        }

        /* Publish packed struct as NT_RAW */
        NT_SetRaw(g_nt_pub, 0, (const uint8_t *)&det_frame, sizeof(det_frame));

        if (det_frame.num_detections > 0) {
            printf("[DS] Frame %u: %u detection(s)  %.1f fps\n",
                   det_frame.frame_number, det_frame.num_detections, g_fps);
        }

        frame_list = frame_list->next;
    }

    return GST_PAD_PROBE_OK;
}

/* ------------------------------------------------------------------ */
/* Bus message handler                                                  */
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
        gchar  *debug = NULL;
        GError *error = NULL;
        gst_message_parse_error(msg, &error, &debug);
        fprintf(stderr, "[GStreamer] Error: %s\n", error->message);
        if (debug)
            fprintf(stderr, "[GStreamer] Debug: %s\n", debug);
        g_error_free(error);
        g_free(debug);
        g_main_loop_quit(loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

/* ------------------------------------------------------------------ */
/* Signal handler for clean shutdown                                    */
/* ------------------------------------------------------------------ */

static void sigint_handler(int sig)
{
    (void)sig;
    if (g_loop)
        g_main_loop_quit(g_loop);
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    GstElement *pipeline, *source, *capsfilter, *tee;
    GstElement *queue_infer, *vidconv, *streammux, *infer, *sink;
    GstElement *queue_web, *vidconv_web, *jpegenc, *appsink_web;
    GstBus     *bus;
    GstPad     *infer_src_pad;
    GstCaps    *caps;

    /* Init GStreamer */
    gst_init(&argc, &argv);
    g_loop = g_main_loop_new(NULL, FALSE);

    /* Init NetworkTables */
    nt_init();

    /* ---- Build pipeline elements ---- */
    pipeline    = gst_pipeline_new("ds-nt-pipeline");
    source      = gst_element_factory_make("v4l2src",       "v4l2-source");
    capsfilter  = gst_element_factory_make("capsfilter",    "caps-filter");
    tee         = gst_element_factory_make("tee",           "vid-tee");

    /* Inference branch */
    queue_infer = gst_element_factory_make("queue",          "queue-infer");
    vidconv     = gst_element_factory_make("nvvideoconvert", "nv-vidconv");
    streammux   = gst_element_factory_make("nvstreammux",    "stream-muxer");
    infer       = gst_element_factory_make("nvinfer",        "primary-nvinference");
    sink        = gst_element_factory_make("fakesink",       "fake-sink");

    /* Web stream branch */
    queue_web   = gst_element_factory_make("queue",          "queue-web");
    vidconv_web = gst_element_factory_make("videoconvert",   "vid-conv-web");
    jpegenc     = gst_element_factory_make("jpegenc",        "jpeg-enc");
    appsink_web = gst_element_factory_make("appsink",        "web-appsink");

    if (!pipeline || !source || !capsfilter || !tee ||
        !queue_infer || !vidconv || !streammux || !infer || !sink ||
        !queue_web || !vidconv_web || !jpegenc || !appsink_web) {
        fprintf(stderr, "[Error] Failed to create GStreamer elements\n");
        return -1;
    }

    /* ---- Configure elements ---- */
    g_object_set(G_OBJECT(source), "device", VIDEO_DEVICE, NULL);

    caps = gst_caps_from_string(
        "video/x-raw,width=" G_STRINGIFY(FRAME_WIDTH)
        ",height=" G_STRINGIFY(FRAME_HEIGHT)
        ",framerate=30/1,format=YUY2");
    g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
    gst_caps_unref(caps);

    g_object_set(G_OBJECT(streammux),
                 "batch-size",           1,
                 "width",                FRAME_WIDTH,
                 "height",               FRAME_HEIGHT,
                 "batched-push-timeout", 40000,
                 "live-source",          TRUE,
                 NULL);

    g_object_set(G_OBJECT(infer),
                 "config-file-path", INFER_CONFIG,
                 NULL);

    g_object_set(G_OBJECT(sink), "sync", FALSE, NULL);

    /* Web branch: 50% quality JPEG, drop old frames so HTTP clients always get fresh ones */
    g_object_set(G_OBJECT(jpegenc), "quality", 50, NULL);
    g_object_set(G_OBJECT(appsink_web),
                 "emit-signals", TRUE,
                 "sync",         FALSE,
                 "max-buffers",  1,
                 "drop",         TRUE,
                 NULL);
    g_signal_connect(appsink_web, "new-sample",
                     G_CALLBACK(on_new_jpeg_sample), NULL);

    /* ---- Assemble pipeline ---- */
    gst_bin_add_many(GST_BIN(pipeline),
                     source, capsfilter, tee,
                     queue_infer, vidconv, streammux, infer, sink,
                     queue_web, vidconv_web, jpegenc, appsink_web,
                     NULL);

    /* v4l2src → capsfilter → tee */
    if (!gst_element_link_many(source, capsfilter, tee, NULL)) {
        fprintf(stderr, "[Error] Failed to link source → capsfilter → tee\n");
        return -1;
    }

    /* tee → queue_infer (GStreamer auto-requests tee src_%u pad) */
    if (!gst_element_link(tee, queue_infer)) {
        fprintf(stderr, "[Error] Failed to link tee → queue_infer\n");
        return -1;
    }

    /* queue_infer → nvvideoconvert */
    if (!gst_element_link(queue_infer, vidconv)) {
        fprintf(stderr, "[Error] Failed to link queue_infer → nvvideoconvert\n");
        return -1;
    }

    /* nvvideoconvert → nvstreammux (request pad) */
    {
        GstPad *sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
        GstPad *srcpad  = gst_element_get_static_pad(vidconv, "src");
        if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
            fprintf(stderr, "[Error] Failed to link vidconv → streammux\n");
            return -1;
        }
        gst_object_unref(srcpad);
        gst_object_unref(sinkpad);
    }

    /* nvstreammux → nvinfer → fakesink */
    if (!gst_element_link_many(streammux, infer, sink, NULL)) {
        fprintf(stderr, "[Error] Failed to link streammux → infer → sink\n");
        return -1;
    }

    /* tee → queue_web → videoconvert → jpegenc → appsink_web */
    if (!gst_element_link(tee, queue_web)) {
        fprintf(stderr, "[Error] Failed to link tee → queue_web\n");
        return -1;
    }
    if (!gst_element_link_many(queue_web, vidconv_web, jpegenc, appsink_web, NULL)) {
        fprintf(stderr, "[Error] Failed to link web branch\n");
        return -1;
    }

    /* ---- Attach pad probe on nvinfer src ---- */
    infer_src_pad = gst_element_get_static_pad(infer, "src");
    if (!infer_src_pad) {
        fprintf(stderr, "[Error] Failed to get nvinfer src pad\n");
        return -1;
    }
    gst_pad_add_probe(infer_src_pad,
                      GST_PAD_PROBE_TYPE_BUFFER,
                      inference_src_pad_buffer_probe,
                      NULL, NULL);
    gst_object_unref(infer_src_pad);

    /* ---- Bus watch ---- */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_watch(bus, bus_call, g_loop);
    gst_object_unref(bus);

    /* ---- Signal handler ---- */
    signal(SIGINT,  sigint_handler);
    signal(SIGTERM, sigint_handler);

    /* ---- Start MJPEG HTTP server thread ---- */
    pthread_t http_tid;
    pthread_create(&http_tid, NULL, http_server_thread, NULL);
    pthread_detach(http_tid);

    /* ---- Start pipeline ---- */
    printf("[DS] Starting pipeline (first run builds TRT engine — please wait)...\n");
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
