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
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
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

/* Camera capture resolution — must match a MJPEG mode the camera reports.
 * Used only in the v4l2src capsfilter.                                     */
#define CAPTURE_WIDTH  1280
#define CAPTURE_HEIGHT 800

/* Internal pipeline resolution fed to nvstreammux (per-stream).
 * 640×400 maintains the camera's native 16:10 (1280×800) aspect ratio.
 * nvinfer's maintain-aspect-ratio=1 + symmetric-padding=1 pads the 640×400
 * input to 640×640 for the model with black bars (120 px top+bottom).       */
#define PIPE_WIDTH  640
#define PIPE_HEIGHT 640

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

/* Camera device paths (populated in main, read by HTTP /set handler) */
static char g_cam_paths[MAX_CAMERAS][32];
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
/* ---- v4l2 camera control ---- */

static void v4l2_set_ctrl(int cam_idx, uint32_t ctrl_id, int value)
{
    int start = (cam_idx < 0) ? 0 : cam_idx;
    int end   = (cam_idx < 0) ? g_num_cameras : cam_idx + 1;
    for (int i = start; i < end; i++) {
        int fd = open(g_cam_paths[i], O_RDWR | O_NONBLOCK);
        if (fd < 0) { fprintf(stderr, "[v4l2] open %s: %m\n", g_cam_paths[i]); continue; }
        
        struct v4l2_control ctrl = { .id = ctrl_id, .value = value };
        if (ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0)
            fprintf(stderr, "[v4l2] VIDIOC_S_CTRL id=%u val=%d on %s: %m\n",
                    ctrl_id, value, g_cam_paths[i]);
        close(fd);
    }
}

static int query_param_int(const char *qs, const char *name, int def)
{
    char needle[40];
    snprintf(needle, sizeof(needle), "%s=", name);
    const char *p = strstr(qs, needle);
    return p ? atoi(p + strlen(needle)) : def;
}

/* ---- Dynamic HTML page (Updated with White Balance) ---- */

static char  g_html_page[10240]; // Increased size for extra HTML
static size_t g_html_len = 0;

static void build_html(int num_cameras)
{
    char opts[128] = "";
    for (int i = 0; i < num_cameras; i++) {
        char opt[32];
        snprintf(opt, sizeof(opt), "<option value='%d'>Cam %d</option>", i, i);
        strncat(opts, opt, sizeof(opts) - strlen(opts) - 1);
    }

    g_html_len = (size_t)snprintf(g_html_page, sizeof(g_html_page),
        "<!DOCTYPE html><html><head><title>DeepStream Vision</title>"
        "<style>"
        "body{margin:0;background:#111;color:#eee;font-family:sans-serif;"
        "display:flex;flex-direction:column;align-items:center;padding:16px;gap:16px;}"
        "img{max-width:100%%;border:2px solid #444;border-radius:4px;}"
        "h2{margin:0;font-size:1.2rem;}"
        ".panel{display:flex;flex-wrap:wrap;gap:20px;background:#1e1e1e;"
        "padding:14px 20px;border-radius:6px;width:100%%;max-width:1000px;box-sizing:border-box;}"
        ".grp{display:flex;flex-direction:column;gap:6px;min-width:180px;}"
        ".grp label{font-size:.82rem;color:#aaa;}"
        "input[type=range]{width:170px;accent-color:#4af;}"
        "select,input[type=checkbox]{accent-color:#4af;}"
        "</style></head>"
        "<body>"
        "<h2>DeepStream YOLOv11n &#8212; Tiled View</h2>"
        "<img src='/stream' alt='Live stream'>"
        "<div class='panel'>"
          "<div class='grp'><label>Camera</label>"
            "<select id='cam'><option value='-1'>All</option>%s</select></div>"
          "<div class='grp'>"
            "<label>Brightness: <b id='bv'>128</b></label>"
            "<input type='range' id='b' min='0' max='255' value='128'></div>"
          "<div class='grp'>"
            "<label><input type='checkbox' id='ae' checked> Auto Exposure</label>"
            "<label>Exposure: <b id='ev'>300</b></label>"
            "<input type='range' id='e' min='1' max='2000' value='300' disabled></div>"
          "<div class='grp'>"
            "<label><input type='checkbox' id='awb' checked> Auto White Balance</label>"
            "<label>Temp (K): <b id='wbv'>4500</b></label>"
            "<input type='range' id='wb' min='2000' max='8000' value='4500' disabled></div>"
        "</div>"
        "<script>"
        "const NC=%d;"
        "const camTargets=()=>{const v=document.getElementById('cam').value;"
          "return v=='-1'?[...Array(NC).keys()]:[parseInt(v)];};"
        "const post=(p)=>camTargets().forEach(c=>fetch('/set?cam='+c+'&'+p));"
        
        "const b=document.getElementById('b');"
        "b.oninput=()=>{document.getElementById('bv').textContent=b.value; post('brightness='+b.value);};"
        
        "const ae=document.getElementById('ae'), e=document.getElementById('e');"
        "ae.onchange=()=>{e.disabled=ae.checked; post('auto_exposure='+(ae.checked?3:1));};"
        "e.oninput=()=>{document.getElementById('ev').textContent=e.value; post('exposure='+e.value);};"
        
        "const awb=document.getElementById('awb'), wb=document.getElementById('wb');"
        "awb.onchange=()=>{wb.disabled=awb.checked; post('auto_wb='+(awb.checked?1:0));};"
        "wb.oninput=()=>{document.getElementById('wbv').textContent=wb.value; post('wb_temp='+wb.value);};"
        "</script>"
        "</body></html>",
        opts, num_cameras);
}

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

    char req[1024] = {0}; // Increased for complex query strings
    recv(fd, req, sizeof(req) - 1, 0);

    if (strstr(req, "GET / ") || strstr(req, "GET /\r")) {
        char hdr[256];
        int hlen = snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            "Content-Length: %zu\r\nConnection: close\r\n\r\n", g_html_len);
        send_all(fd, hdr, (size_t)hlen);
        send_all(fd, g_html_page, g_html_len);

    } else if (strstr(req, "GET /set?")) {
        char *qs = strstr(req, "/set?") + 5;
        int cam  = query_param_int(qs, "cam", -1);

        // Brightness
        if (strstr(qs, "brightness="))
            v4l2_set_ctrl(cam, V4L2_CID_BRIGHTNESS, query_param_int(qs, "brightness", 128));
        
        // Exposure Logic
        if (strstr(qs, "auto_exposure="))
            v4l2_set_ctrl(cam, V4L2_CID_EXPOSURE_AUTO, query_param_int(qs, "auto_exposure", 3));
        if (strstr(qs, "exposure="))
            v4l2_set_ctrl(cam, V4L2_CID_EXPOSURE_ABSOLUTE, query_param_int(qs, "exposure", 300));
            
        // White Balance Logic
        if (strstr(qs, "auto_wb="))
            v4l2_set_ctrl(cam, V4L2_CID_AUTO_WHITE_BALANCE, query_param_int(qs, "auto_wb", 1));
        if (strstr(qs, "wb_temp="))
            v4l2_set_ctrl(cam, V4L2_CID_WHITE_BALANCE_TEMPERATURE, query_param_int(qs, "wb_temp", 4500));

        send_all(fd, "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n", 47);

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
            gsize fsz = g_jpeg.size;
            g_jpeg_ready = FALSE;
            pthread_mutex_unlock(&g_jpeg_mutex);

            char part[256];
            int plen = snprintf(part, sizeof(part),
                "--" MJPEG_BOUNDARY "\r\nContent-Type: image/jpeg\r\n"
                "Content-Length: %zu\r\n\r\n", fsz);
            
            if (send(fd, part, (size_t)plen, MSG_NOSIGNAL) <= 0 ||
                send(fd, frame, fsz, MSG_NOSIGNAL) <= 0 ||
                send(fd, "\r\n", 2, MSG_NOSIGNAL) <= 0) {
                g_free(frame); break;
            }
            g_free(frame);
        }
    } else {
        send_all(fd, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", 72);
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
    printf("[HTTP] Tiled MJPEG stream -> http://<jetson-ip>:%d/\n", MJPEG_PORT);

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
static GstPadProbeReturn
inference_src_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    // 1. ATOMIC LOCK: If the probe is already running on this thread/buffer, 
    // exit immediately. This prevents recursive "spinning" loops.
    static volatile int in_probe = 0;
    if (__sync_lock_test_and_set(&in_probe, 1)) {
        return GST_PAD_PROBE_OK; 
    }

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) {
        __sync_lock_release(&in_probe);
        return GST_PAD_PROBE_OK;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        __sync_lock_release(&in_probe);
        return GST_PAD_PROBE_OK;
    }

    int64_t now = now_us();
    static int global_frame_count = 0;
    global_frame_count++;

    // 2. STRICT ITERATION: We use num_frames_in_batch as the hard limit.
    // This ignores any corrupted "next" pointers that point back to the start.
    NvDsFrameMetaList *fl = batch_meta->frame_meta_list;
    for (int i = 0; i < (int)batch_meta->num_frames_in_batch && fl != NULL; i++) {
        NvDsFrameMeta *fm = (NvDsFrameMeta *)fl->data;
        if (!fm) break;

        int src = (int)fm->source_id;
        if (src < 0 || src >= g_num_cameras) {
            fl = fl->next;
            continue;
        }

        // Only print every 100 batches to save CPU cycles
        if (global_frame_count % 100 == 0) {
            printf("[DS] Processing Source %d | Batch Frame %d/%d\n", 
                   src, i + 1, batch_meta->num_frames_in_batch);
        }

        /* --- FPS CALCULATION --- */
        if (g_last_ts[src] > 0) {
            float inst = 1000000.0f / (float)(now - g_last_ts[src]);
            g_fps[src] = (g_fps[src] == 0.0f) ? inst : (FPS_ALPHA * inst + (1.0f - FPS_ALPHA) * g_fps[src]);
        }
        g_last_ts[src] = now;

        /* --- DETECTION DATA (NetworkTables) --- */
        DetectionFrame det = {0};
        det.frame_number = (uint32_t)fm->frame_num;
        det.timestamp_us = (uint64_t)now;
        det.fps          = g_fps[src];
        det.source_id    = (uint32_t)src;

        float fw = fm->source_frame_width  > 0 ? (float)fm->source_frame_width  : 1280.0f;
        float fh = fm->source_frame_height > 0 ? (float)fm->source_frame_height : 800.0f;

        // Object Loop with Safety Limit
        // Object Loop with Safety Limit
        int obj_count = 0;
        for (NvDsObjectMetaList *ol = fm->obj_meta_list; ol != NULL && obj_count < 100; ol = ol->next) {
            obj_count++;
            NvDsObjectMeta *om = (NvDsObjectMeta *)ol->data;
            
            // Only process and display if confidence is above 0.95
            if (om && det.num_detections < MAX_DETECTIONS && om->confidence > 0.95) {
                
                /* 1. Update NetworkTables / Logic Data */
                Detection *d = &det.detections[det.num_detections++];
                d->class_id   = om->class_id;
                d->confidence = om->confidence;
                d->left       = om->rect_params.left   / fw;
                d->top        = om->rect_params.top    / fh;
                d->width      = om->rect_params.width  / fw;
                d->height     = om->rect_params.height / fh;
                g_strlcpy(d->label, om->obj_label, sizeof(d->label));

                /* 2. Update the On-Screen Label with Confidence */
                // DeepStream allocates a default string; we replace it with our own
                // Format: "fuel 97%"
                char new_label[64];
                snprintf(new_label, sizeof(new_label), "%s %.0f%%", 
                         om->obj_label, om->confidence * 100.0);

                // Free the old text if it exists to avoid a memory leak, then assign new
                if (om->text_params.display_text) {
                    g_free(om->text_params.display_text);
                }
                om->text_params.display_text = g_strdup(new_label);

                // Optional: Force the background color so it's readable
                om->text_params.set_bg_clr = 1;
                om->text_params.text_bg_clr = (NvOSD_ColorParams){0.0, 0.0, 0.0, 0.5}; 
            }
        }

        // Send to RoboRIO (Commented out for initial spin test)
        NT_SetRaw(g_nt_pub[src], 0, (const uint8_t *)&det, sizeof(det));

        /* --- OSD OVERLAY --- */
        NvDsDisplayMeta *dm = nvds_acquire_display_meta_from_pool(batch_meta);
        if (dm) {
            dm->num_labels = 1;
            NvOSD_TextParams *tp = &dm->text_params[0];
            tp->display_text = (char *)g_malloc0(64);
            snprintf(tp->display_text, 63, "CAM %d | %.1f FPS", src, g_fps[src]);
            tp->x_offset = 20; tp->y_offset = 20;
            tp->font_params.font_name = (char *)"Sans";
            tp->font_params.font_size = 12;
            tp->font_params.font_color = (NvOSD_ColorParams){1.0, 1.0, 1.0, 1.0};
            tp->set_bg_clr = 1;
            tp->text_bg_clr = (NvOSD_ColorParams){0.0, 0.0, 0.0, 0.5};
            nvds_add_display_meta_to_frame(fm, dm);
        }

        // Move to next frame in list
        NvDsFrameMetaList *next_node = fl->next;
        if (next_node == fl) break; // Final check for circular self-reference
        fl = next_node;
    }

    // 3. RELEASE LOCK: Allow the next buffer to be processed
    __sync_lock_release(&in_probe);
    return GST_PAD_PROBE_OK;
}

/* ------------------------------------------------------------------ */
/* Dynamic pad callbacks                                                 */
/* ------------------------------------------------------------------ */

/* nvurisrcbin → nvstreammux (file mode only) */
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

    /* Copy device paths to global so HTTP /set handler can reach them */
    for (int i = 0; i < g_num_cameras; i++)
        strncpy(g_cam_paths[i], camera_device[i], sizeof(g_cam_paths[0]) - 1);

    nt_init(g_num_cameras);
    build_html(g_num_cameras);

    /* ---- Tiler layout: 1 row × N columns ---- */
    int tiler_cols = g_num_cameras;
    int tiler_rows = 1;
    int tiled_w    = PIPE_WIDTH * g_num_cameras;  /* total output width  */
    int tiled_h    = PIPE_HEIGHT;                 /* total output height */

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
    GstElement *source   [MAX_CAMERAS] = {NULL};
    GstElement *caps_f   [MAX_CAMERAS] = {NULL};
    GstElement *src_queue[MAX_CAMERAS] = {NULL};
    GstElement *jpegdec  [MAX_CAMERAS] = {NULL};
    GstElement *cpuconv  [MAX_CAMERAS] = {NULL};  /* CPU videoconvert: I420→NV12        */
    GstElement *vidconv  [MAX_CAMERAS] = {NULL};  /* nvvideoconvert: CPU NV12→NVMM NV12 */

    if (use_file) {
        source[0] = gst_element_factory_make("nvurisrcbin", "uri-source");
        if (!source[0]) { fprintf(stderr, "[Error] Could not create nvurisrcbin\n"); return -1; }
        g_object_set(source[0], "uri", video_uri, "gpu-id", 0, NULL);
        g_signal_connect(source[0], "pad-added", G_CALLBACK(on_pad_added), streammux);
    } else {
        for (int i = 0; i < g_num_cameras; i++) {
            char src_name[32], caps_name[32], q_name[32], dec_name[32], cc_name[32], vc_name[32];
            snprintf(src_name,  sizeof(src_name),  "v4l2-src-%d",    i);
            snprintf(caps_name, sizeof(caps_name), "caps-filter-%d", i);
            snprintf(q_name,    sizeof(q_name),    "src-queue-%d",   i);
            snprintf(dec_name,  sizeof(dec_name),  "jpegdec-%d",     i);
            snprintf(cc_name,   sizeof(cc_name),   "cpuconv-%d",     i);
            snprintf(vc_name,   sizeof(vc_name),   "src-nvconv-%d",  i);

            source   [i] = gst_element_factory_make("v4l2src",        src_name);
            caps_f   [i] = gst_element_factory_make("capsfilter",      caps_name);
            /* queue decouples v4l2src from the CPU decoder */
            src_queue[i] = gst_element_factory_make("queue",           q_name);
            /* CPU JPEG decoder — outputs I420/YUY2 in system memory.
             * Using CPU path avoids NvBufSurfTransform (VIC) which fails when
             * converting between NVMM memory types on this Jetson.             */
            jpegdec  [i] = gst_element_factory_make("jpegdec",         dec_name);
            /* CPU videoconvert normalises to NV12 before nvvideoconvert */
            cpuconv  [i] = gst_element_factory_make("videoconvert",    cc_name);
            /* nvvideoconvert: CPU NV12 → NVMM NV12.  Because the input is
             * system memory (not NVMM), this uses a direct DMA copy instead
             * of NvBufSurfTransform, which is the unsupported path.            */
            vidconv  [i] = gst_element_factory_make("nvvideoconvert",  vc_name);

            if (!source[i] || !caps_f[i] || !src_queue[i] ||
                !jpegdec[i] || !cpuconv[i] || !vidconv[i]) {
                fprintf(stderr, "[Error] Failed to create elements for camera %d\n", i);
                return -1;
            }

            g_object_set(source[i], "device", camera_device[i], NULL);
            /* copy-hw=2: GPU/CUDA path — see shared element comment above */
            g_object_set(vidconv[i], "copy-hw", 2, NULL);
            /* Leaky downstream queue: drop oldest compressed frame when nvinfer
             * is busy.  Camera runs at 120fps; inference at ~30fps.  Without
             * this the queue fills, NVMM DMA fds accumulate, and CUDA mappings
             * eventually become invalid.                                       */
            g_object_set(src_queue[i],
                         "leaky",            2,   /* GST_QUEUE_LEAK_DOWNSTREAM */
                         "max-size-buffers", 4,
                         "max-size-bytes",   0,
                         "max-size-time",    0,
                         NULL);
            printf("[DS]   camera %d → %s\n", i, camera_device[i]);

            /* Lock v4l2src to the camera's native MJPEG mode (1280×800 @ 120 fps). */
            GstCaps *caps = gst_caps_from_string(
                "image/jpeg,width=" G_STRINGIFY(CAPTURE_WIDTH)
                ",height=" G_STRINGIFY(CAPTURE_HEIGHT)
                ",framerate=120/1");
            g_object_set(caps_f[i], "caps", caps, NULL);
            gst_caps_unref(caps);
        }
    }

    /* ---- Configure shared elements ---- */
    g_object_set(streammux,
                 "batch-size",           g_num_cameras,
                 "width",                PIPE_WIDTH,
                 "height",               PIPE_HEIGHT,
                 "batched-push-timeout", 100000,
                 "live-source",          use_file ? FALSE : TRUE,
                 /* compute-hw=1 → GPU/CUDA path for all VIC transforms inside
                  * nvstreammux.  Avoids the VIC 16×16-min bug in DS 7.1 / JP 6.x
                  * that causes deferred cudaErrorIllegalAddress after ~30 s.    */
                 "compute-hw",           1,
                 NULL);
    g_object_set(infer,   "config-file-path", INFER_CONFIG, NULL);
    g_object_set(nvdsosd, "process-mode",     0,            NULL);

    /* copy-hw=2 → CUDA path for ALL nvvideoconvert elements.
     * On Jetson Orin + DeepStream 7.1 the default VIC path in nvvideoconvert
     * corrupts DMA buffer CUDA mappings after tens of seconds of operation,
     * causing cudaErrorIllegalAddress 700 inside nvinfer preprocessing.
     * copy-hw=2 (GPU) is the NVIDIA-confirmed workaround (DS forums #332306,
     * #335930, #364425).  Apply to every nvvideoconvert in the pipeline.    */
    g_object_set(vidconv_osd, "copy-hw", 2, NULL);
    g_object_set(nv2cpu,      "copy-hw", 2, NULL);
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
    /* Correct Order: Infer -> Convert -> Tiler -> OSD */
    gst_bin_add_many(GST_BIN(pipeline),
                    streammux, infer, 
                    vidconv_osd, tiler, // Tiler comes BEFORE OSD
                    nvdsosd, tee, 
                    queue_inf, sink,
                    queue_web, nv2cpu, vcpucpu, jpegenc, appsink,
                    NULL);

    if (use_file) {
        gst_bin_add(GST_BIN(pipeline), source[0]);
    } else {
        for (int i = 0; i < g_num_cameras; i++)
            gst_bin_add_many(GST_BIN(pipeline),
                             source[i], caps_f[i], src_queue[i],
                             jpegdec[i], cpuconv[i], vidconv[i], NULL);
    }

    /* ---- Link sources → streammux ---- */
    if (!use_file) {
        for (int i = 0; i < g_num_cameras; i++) {
            /* v4l2src → capsfilter(MJPEG) → queue → jpegdec → videoconvert → nvvideoconvert */
            if (!gst_element_link_many(source[i], caps_f[i], src_queue[i],
                                       jpegdec[i], cpuconv[i], vidconv[i], NULL)) {
                fprintf(stderr, "[Error] Failed to link camera %d source chain\n", i);
                return -1;
            }
            /* nvvideoconvert src → streammux sink_N */
            char pad_name[16];
            snprintf(pad_name, sizeof(pad_name), "sink_%d", i);
            if (!gst_element_link_pads(vidconv[i], "src", streammux, pad_name)) {
                fprintf(stderr, "[Error] Failed to link vidconv[%d] → streammux\n", i);
                return -1;
            }
        }
    }
    /* file mode: nvurisrcbin → streammux via on_pad_added callback */

    /* ---- Link shared inference + display path ---- */
    if (!gst_element_link(streammux, infer)) {
        fprintf(stderr, "[Error] Failed to link streammux → infer\n");
        return -1;
    }

    /* CORRECT ORDER: infer → vidconv_osd → tiler → nvdsosd → tee */
    /* 1. Convert to RGBA for the Tiler/OSD */
    GstCaps *rgba_caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=RGBA");

    if (!gst_element_link(infer, vidconv_osd) ||
        !gst_element_link_filtered(vidconv_osd, tiler, rgba_caps)) { // Link TO TILER first
        fprintf(stderr, "[Error] Failed to link infer → vidconv → tiler\n");
        return -1;
    }

    /* 2. Link Tiler → OSD → Tee */
    if (!gst_element_link_many(tiler, nvdsosd, tee, NULL)) {
        fprintf(stderr, "[Error] Failed to link tiler → osd → tee\n");
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
