/*
 * ds_nt_probe.c
 *
 * Multi-camera DeepStream YOLOv11n pipeline with NetworkTables publisher
 * and tiled MJPEG HTTP stream.
 *
 * Camera mode  (VIDEO_URI not set):
 *   v4l2src[0..N-1] → nvvideoconvert[i] → nvstreammux (batch=N)
 *                                            → nvinfer
 *                                            → nvvideoconvert (NV12→RGBA)
 *                                            → nvdsosd   (draws bboxes per stream)
 *                                            → nvmultistreamtiler (tiles into 1 frame)
 *                                            → tee ─┬→ fakesink
 *                                                    └→ nvvideoconvert → videoconvert
 *                                                       → jpegenc → appsink → HTTP :8080
 *
 *   Pad probe on nvinfer src publishes DetectionFrame per source to
 *   /vision/detections/0, /vision/detections/1, ...
 *
 * File mode  (VIDEO_URI set, single-source testing):
 *   nvurisrcbin ─(pad-added)→ nvstreammux → ... (same from streammux onward)
 *
 * Environment:
 *   NUM_CAMERAS   number of USB cameras (default 2, camera mode only)
 *   VIDEO_URI     if set, uses file/RTSP source instead of cameras
 *   NT_TEAM_NUMBER  FRC team number
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

#define INFER_CONFIG   "/opt/deepstream/config/config_infer_primary_yolo11n.txt"
#define VIDEO_DEVICE_FMT "/dev/video%d"
#define MAX_CAMERAS    4

/* ------------------------------------------------------------------ */
/* Globals                                                              */
/* ------------------------------------------------------------------ */

static GMainLoop    *g_loop         = NULL;
static NT_Inst       g_nt_inst      = 0;
static NT_Publisher  g_nt_pub[MAX_CAMERAS];
static int           g_num_cameras  = 1;

/* Per-source FPS tracking */
static float    g_fps[MAX_CAMERAS]     = {0};
static int64_t  g_last_ts[MAX_CAMERAS] = {0};
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
    "<body><h2>DeepStream YOLOv11n — Tiled View</h2>"
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
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(MJPEG_PORT);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[HTTP] bind"); close(srv); return NULL;
    }
    listen(srv, 10);
    printf("[HTTP] Tiled MJPEG stream →  http://<jetson-ip>:%d/\n", MJPEG_PORT);

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

static void nt_init(int num_cameras)
{
    const char *team_str = getenv("NT_TEAM_NUMBER");
    unsigned int team    = team_str ? (unsigned int)atoi(team_str) : 0;

    struct WPI_String identity = { "jetson-vision", 13 };
    struct WPI_String type_str = { "raw",            3 };
    struct WPI_String fallback = { "roborio-0-frc.local", 19 };

    g_nt_inst = NT_GetDefaultInstance();
    NT_StartClient4(g_nt_inst, &identity);

    if (team > 0) {
        printf("[NT] Connecting to team %u roboRIO\n", team);
        NT_SetServerTeam(g_nt_inst, team, 0);
    } else {
        fprintf(stderr, "[NT] Warning: NT_TEAM_NUMBER not set, trying roborio-0-frc.local\n");
        NT_SetServer(g_nt_inst, &fallback, 1735);
    }

    struct NT_PubSubOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.structSize = sizeof(opts);

    for (int i = 0; i < num_cameras; i++) {
        char path[64];
        snprintf(path, sizeof(path), "/vision/detections/%d", i);
        struct WPI_String topic_name = { path, strlen(path) };
        NT_Topic topic = NT_GetTopic(g_nt_inst, &topic_name);
        g_nt_pub[i] = NT_Publish(topic, NT_RAW, &type_str, &opts);
        printf("[NT] Publisher created for %s\n", path);
    }
}

/* ------------------------------------------------------------------ */
/* Pad probe (nvinfer src) — publishes per-source detections            */
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

    NvDsFrameMetaList *fl = batch_meta->frame_meta_list;
    while (fl) {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)fl->data;
        if (!fm) { fl = fl->next; continue; }

        int src = (int)fm->source_id;
        if (src < 0 || src >= g_num_cameras) { fl = fl->next; continue; }

        /* Per-source FPS */
        if (g_last_ts[src] > 0) {
            float inst = 1000000.0f / (float)(now - g_last_ts[src]);
            g_fps[src] = (g_fps[src] == 0.0f)
                       ? inst : FPS_ALPHA * inst + (1.0f - FPS_ALPHA) * g_fps[src];
        }
        g_last_ts[src] = now;

        DetectionFrame det = {0};
        det.frame_number = (uint32_t)(fm->frame_num);
        det.timestamp_us = now;
        det.fps          = g_fps[src];
        det.source_id    = (uint32_t)src;

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

        NT_SetRaw(g_nt_pub[src], 0, (const uint8_t *)&det, sizeof(det));

        /* FPS overlay: add display text before nvdsosd renders it */
        NvDsDisplayMeta *dm = nvds_acquire_display_meta_from_pool(batch_meta);
        if (dm) {
            dm->num_labels = 1;
            NvOSD_TextParams *tp = &dm->text_params[0];
            tp->display_text = (char *)g_malloc(32);
            snprintf(tp->display_text, 32, "cam%d  %.1f fps", src, g_fps[src]);
            tp->x_offset = 8;
            tp->y_offset = 8;
            tp->font_params.font_name  = "Sans";
            tp->font_params.font_size  = 14;
            tp->font_params.font_color = (NvOSD_ColorParams){1.0, 1.0, 1.0, 1.0};
            tp->set_bg_clr  = 1;
            tp->text_bg_clr = (NvOSD_ColorParams){0.0, 0.0, 0.0, 0.6};
            nvds_add_display_meta_to_frame(fm, dm);
        }

        if (det.num_detections > 0)
            printf("[DS] cam%d frame %u: %u detection(s)  %.1f fps\n",
                   src, det.frame_number, det.num_detections, g_fps[src]);

        fl = fl->next;
    }
    return GST_PAD_PROBE_OK;
}

/* ------------------------------------------------------------------ */
/* Dynamic pad callback (nvurisrcbin → nvstreammux, file mode only)     */
/* ------------------------------------------------------------------ */

static void on_pad_added(GstElement *src, GstPad *new_pad, gpointer data)
{
    (void)src;
    GstElement *streammux = (GstElement *)data;

    GstCaps *caps = gst_pad_get_current_caps(new_pad);
    if (!caps) caps = gst_pad_query_caps(new_pad, NULL);
    const gchar *mime = gst_structure_get_name(gst_caps_get_structure(caps, 0));
    gboolean is_video = g_str_has_prefix(mime, "video/");
    gst_caps_unref(caps);
    if (!is_video) return;

    GstPad *sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    if (!sinkpad) { fprintf(stderr, "[Error] Could not get streammux sink_0\n"); return; }
    if (gst_pad_link(new_pad, sinkpad) != GST_PAD_LINK_OK)
        fprintf(stderr, "[Error] Failed to link uri-source → streammux\n");
    gst_object_unref(sinkpad);
}

/* ------------------------------------------------------------------ */
/* Bus handler + signal                                                  */
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

    /* ---- Source mode ---- */
    const char *video_uri = getenv("VIDEO_URI");
    if (!video_uri) video_uri = "";
    gboolean use_file = (video_uri[0] != '\0');

    if (use_file) {
        g_num_cameras = 1;
        printf("[DS] Source: %s\n", video_uri);
    } else {
        const char *nc_str = getenv("NUM_CAMERAS");
        g_num_cameras = nc_str ? atoi(nc_str) : 2;
        if (g_num_cameras < 1) g_num_cameras = 1;
        if (g_num_cameras > MAX_CAMERAS) g_num_cameras = MAX_CAMERAS;
        printf("[DS] Source: %d USB camera(s)\n", g_num_cameras);
    }

    /* Parse CAMERA_DEVICES=  /dev/video0,/dev/video2  (overrides /dev/videoN default) */
    char camera_device[MAX_CAMERAS][32];
    const char *cam_devs_env = getenv("CAMERA_DEVICES");
    if (cam_devs_env && cam_devs_env[0] != '\0') {
        char buf[256];
        strncpy(buf, cam_devs_env, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        int idx = 0;
        char *tok = strtok(buf, ",");
        while (tok && idx < MAX_CAMERAS) {
            strncpy(camera_device[idx], tok, sizeof(camera_device[0]) - 1);
            camera_device[idx][sizeof(camera_device[0]) - 1] = '\0';
            idx++;
            tok = strtok(NULL, ",");
        }
        /* Fill remaining with defaults if CAMERA_DEVICES has fewer entries than NUM_CAMERAS */
        for (int i = idx; i < g_num_cameras; i++)
            snprintf(camera_device[i], sizeof(camera_device[0]), VIDEO_DEVICE_FMT, i * 2);
    } else {
        /* Default: /dev/video0, /dev/video2, /dev/video4 ... (skip metadata nodes) */
        for (int i = 0; i < g_num_cameras; i++)
            snprintf(camera_device[i], sizeof(camera_device[0]), VIDEO_DEVICE_FMT, i * 2);
    }

    nt_init(g_num_cameras);

    /* ---- Tiler layout: 1 row × N columns ---- */
    int tiler_cols = g_num_cameras;
    int tiler_rows = 1;
    int tiled_w    = FRAME_WIDTH * g_num_cameras;  /* total output width  */
    int tiled_h    = FRAME_HEIGHT;                 /* total output height */

    /* ---- Shared pipeline elements ---- */
    GstElement *pipeline    = gst_pipeline_new("ds-nt-pipeline");
    GstElement *streammux   = gst_element_factory_make("nvstreammux",        "stream-muxer");
    GstElement *infer       = gst_element_factory_make("nvinfer",            "primary-nvinference");
    GstElement *vidconv_osd = gst_element_factory_make("nvvideoconvert",     "nv-vidconv-osd");
    GstElement *nvdsosd     = gst_element_factory_make("nvdsosd",            "nv-onscreendisplay");
    GstElement *tiler       = gst_element_factory_make("nvmultistreamtiler", "nv-tiler");
    GstElement *tee         = gst_element_factory_make("tee",                "post-tiler-tee");
    GstElement *queue_inf   = gst_element_factory_make("queue",              "queue-infer");
    GstElement *sink        = gst_element_factory_make("fakesink",           "fake-sink");
    GstElement *queue_web   = gst_element_factory_make("queue",              "queue-web");
    GstElement *nv2cpu      = gst_element_factory_make("nvvideoconvert",     "nv-to-cpu");
    GstElement *vcpucpu     = gst_element_factory_make("videoconvert",       "cpu-conv");
    GstElement *jpegenc     = gst_element_factory_make("jpegenc",            "jpeg-enc");
    GstElement *appsink     = gst_element_factory_make("appsink",            "web-appsink");

    if (!pipeline || !streammux || !infer || !vidconv_osd || !nvdsosd ||
        !tiler || !tee || !queue_inf || !sink ||
        !queue_web || !nv2cpu || !vcpucpu || !jpegenc || !appsink) {
        fprintf(stderr, "[Error] Failed to create shared pipeline elements\n");
        return -1;
    }

    /* ---- Per-source elements (camera mode) ---- */
    GstElement *source  [MAX_CAMERAS] = {NULL};
    GstElement *caps_f  [MAX_CAMERAS] = {NULL};
    GstElement *vidconv [MAX_CAMERAS] = {NULL};

    if (use_file) {
        source[0] = gst_element_factory_make("nvurisrcbin", "uri-source");
        if (!source[0]) { fprintf(stderr, "[Error] Could not create nvurisrcbin\n"); return -1; }
        g_object_set(source[0], "uri", video_uri, "gpu-id", 0, NULL);
        g_signal_connect(source[0], "pad-added", G_CALLBACK(on_pad_added), streammux);
    } else {
        for (int i = 0; i < g_num_cameras; i++) {
            char src_name[32], caps_name[32], vc_name[32];
            snprintf(src_name,  sizeof(src_name),  "v4l2-src-%d",    i);
            snprintf(caps_name, sizeof(caps_name), "caps-filter-%d", i);
            snprintf(vc_name,   sizeof(vc_name),   "nv-vidconv-%d",  i);

            source [i] = gst_element_factory_make("v4l2src",        src_name);
            caps_f [i] = gst_element_factory_make("capsfilter",      caps_name);
            vidconv[i] = gst_element_factory_make("nvvideoconvert",  vc_name);

            if (!source[i] || !caps_f[i] || !vidconv[i]) {
                fprintf(stderr, "[Error] Failed to create elements for camera %d\n", i);
                return -1;
            }

            g_object_set(source[i], "device", camera_device[i], NULL);
            printf("[DS]   camera %d → %s\n", i, camera_device[i]);

            /* Let v4l2src negotiate its native resolution/framerate.
             * nvstreammux will scale to FRAME_WIDTH×FRAME_HEIGHT anyway. */
            GstCaps *caps = gst_caps_from_string("video/x-raw");
            g_object_set(caps_f[i], "caps", caps, NULL);
            gst_caps_unref(caps);
        }
    }

    /* ---- Configure shared elements ---- */
    g_object_set(streammux,
                 "batch-size",           g_num_cameras,
                 "width",                FRAME_WIDTH,   /* per-stream; tiler scales to tiled_w */
                 "height",               FRAME_HEIGHT,
                 "batched-push-timeout", 40000,
                 "live-source",          use_file ? FALSE : TRUE,
                 NULL);
    g_object_set(infer,   "config-file-path", INFER_CONFIG, NULL);
    g_object_set(nvdsosd, "process-mode",     0,            NULL);
    g_object_set(tiler,
                 "rows",    tiler_rows,
                 "columns", tiler_cols,
                 "width",   tiled_w,
                 "height",  tiled_h,
                 NULL);
    g_object_set(sink,    "sync", FALSE, NULL);
    g_object_set(jpegenc, "quality", 50,  NULL);
    g_object_set(appsink,
                 "emit-signals", TRUE, "sync", FALSE,
                 "max-buffers",  1,    "drop", TRUE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_jpeg_sample), NULL);

    /* ---- Add all elements to pipeline bin ---- */
    gst_bin_add_many(GST_BIN(pipeline),
                     streammux, infer, vidconv_osd, nvdsosd, tiler, tee,
                     queue_inf, sink,
                     queue_web, nv2cpu, vcpucpu, jpegenc, appsink,
                     NULL);

    if (use_file) {
        gst_bin_add(GST_BIN(pipeline), source[0]);
    } else {
        for (int i = 0; i < g_num_cameras; i++)
            gst_bin_add_many(GST_BIN(pipeline), source[i], caps_f[i], vidconv[i], NULL);
    }

    /* ---- Link sources → streammux ---- */
    if (!use_file) {
        for (int i = 0; i < g_num_cameras; i++) {
            /* v4l2src → capsfilter → nvvideoconvert */
            if (!gst_element_link_many(source[i], caps_f[i], vidconv[i], NULL)) {
                fprintf(stderr, "[Error] Failed to link camera %d source chain\n", i);
                return -1;
            }
            /* nvvideoconvert → streammux sink_N */
            char pad_name[16];
            snprintf(pad_name, sizeof(pad_name), "sink_%d", i);
            GstPad *mux_sink = gst_element_request_pad_simple(streammux, pad_name);
            GstPad *vc_src   = gst_element_get_static_pad(vidconv[i], "src");
            if (gst_pad_link(vc_src, mux_sink) != GST_PAD_LINK_OK) {
                fprintf(stderr, "[Error] Failed to link vidconv[%d] → streammux\n", i);
                return -1;
            }
            gst_object_unref(vc_src);
            gst_object_unref(mux_sink);
        }
    }
    /* file mode: nvurisrcbin → streammux via on_pad_added callback */

    /* ---- Link shared inference + display path ---- */
    if (!gst_element_link_many(streammux, infer, NULL)) {
        fprintf(stderr, "[Error] Failed to link streammux → infer\n");
        return -1;
    }

    /* nvinfer → nvvideoconvert (NV12→RGBA NVMM) → nvdsosd → tiler → tee */
    GstCaps *rgba_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");
    if (!gst_element_link(infer, vidconv_osd) ||
        !gst_element_link_filtered(vidconv_osd, nvdsosd, rgba_caps) ||
        !gst_element_link_many(nvdsosd, tiler, tee, NULL)) {
        fprintf(stderr, "[Error] Failed to link infer→osd→tiler→tee\n");
        gst_caps_unref(rgba_caps);
        return -1;
    }
    gst_caps_unref(rgba_caps);

    /* tee → queue_inf → fakesink */
    if (!gst_element_link_many(tee, queue_inf, sink, NULL)) {
        fprintf(stderr, "[Error] Failed to link tee → fakesink branch\n");
        return -1;
    }

    /* tee → queue_web → nv2cpu (RGBA NVMM → CPU I420) → videoconvert → jpegenc → appsink */
    GstCaps *cpu_caps = gst_caps_from_string("video/x-raw,format=I420");
    if (!gst_element_link(tee, queue_web) ||
        !gst_element_link(queue_web, nv2cpu) ||
        !gst_element_link_filtered(nv2cpu, vcpucpu, cpu_caps) ||
        !gst_element_link_many(vcpucpu, jpegenc, appsink, NULL)) {
        fprintf(stderr, "[Error] Failed to link web branch\n");
        gst_caps_unref(cpu_caps);
        return -1;
    }
    gst_caps_unref(cpu_caps);

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
    printf("[DS] Starting pipeline (%s, batch=%d)...\n",
           use_file ? "file" : "camera", g_num_cameras);
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
