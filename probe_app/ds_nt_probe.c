/*
 * ds_nt_probe.c
 *
 * DeepStream YOLOv11n inference pipeline with NetworkTables publisher.
 *
 * Pipeline:
 *   v4l2src → capsfilter → nvvideoconvert → nvstreammux
 *     → nvinfer → fakesink
 *                  ^-- pad probe extracts NvDsBatchMeta and publishes
 *                      DetectionFrame structs as NT_RAW to /vision/detections
 *
 * Build: see CMakeLists.txt
 * Run:   NT_TEAM_NUMBER=XXXX ./ds_nt_probe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <gst/gst.h>
#include <glib.h>

/* DeepStream metadata headers (from /opt/nvidia/deepstream/deepstream/sources/includes) */
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

/* WPILib NetworkTables C API */
#include <networktables/ntcore.h>

#include "detection_types.h"

/* ------------------------------------------------------------------ */
/* Globals                                                              */
/* ------------------------------------------------------------------ */

static GMainLoop  *g_loop     = NULL;
static NT_Inst     g_nt_inst  = 0;
static NT_Publisher g_nt_pub  = 0;
static guint32     g_frame_no = 0;

#define INFER_CONFIG "/opt/deepstream/config/config_infer_primary_yolo11n.txt"
#define VIDEO_DEVICE "/dev/video0"

/* ------------------------------------------------------------------ */
/* NetworkTables setup                                                  */
/* ------------------------------------------------------------------ */

static void nt_init(void)
{
    const char *team_str = getenv("NT_TEAM_NUMBER");
    unsigned int team    = team_str ? (unsigned int)atoi(team_str) : 0;

    g_nt_inst = NT_GetDefaultInstance();
    NT_StartClient4(g_nt_inst, "jetson-vision");

    if (team > 0) {
        printf("[NT] Connecting to team %u roboRIO\n", team);
        NT_SetServerTeam(g_nt_inst, team, 0);  /* 0 = default port 1735 */
    } else {
        /* Fallback: connect to roborio-0-frc.local if no team set */
        fprintf(stderr, "[NT] Warning: NT_TEAM_NUMBER not set. Trying roborio-0-frc.local\n");
        NT_SetServer(g_nt_inst, "roborio-0-frc.local", 1735);
    }

    NT_Topic topic = NT_GetTopic(g_nt_inst, "/vision/detections");
    g_nt_pub = NT_Publish(topic, NT_RAW, "raw", NULL, 0);

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
            printf("[DS] Frame %u: %u detection(s)\n",
                   det_frame.frame_number, det_frame.num_detections);
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
    GstElement *pipeline, *source, *capsfilter, *vidconv,
               *streammux, *infer, *sink;
    GstBus     *bus;
    GstPad     *infer_src_pad;
    GstCaps    *caps;

    /* Init GStreamer */
    gst_init(&argc, &argv);
    g_loop = g_main_loop_new(NULL, FALSE);

    /* Init NetworkTables */
    nt_init();

    /* ---- Build pipeline elements ---- */
    pipeline   = gst_pipeline_new("ds-nt-pipeline");
    source     = gst_element_factory_make("v4l2src",       "v4l2-source");
    capsfilter = gst_element_factory_make("capsfilter",    "caps-filter");
    vidconv    = gst_element_factory_make("nvvideoconvert", "nv-vidconv");
    streammux  = gst_element_factory_make("nvstreammux",   "stream-muxer");
    infer      = gst_element_factory_make("nvinfer",       "primary-nvinference");
    sink       = gst_element_factory_make("fakesink",      "fake-sink");

    if (!pipeline || !source || !capsfilter || !vidconv ||
        !streammux || !infer || !sink) {
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

    /* ---- Assemble pipeline ---- */
    gst_bin_add_many(GST_BIN(pipeline),
                     source, capsfilter, vidconv, streammux, infer, sink, NULL);

    /* v4l2src → capsfilter → nvvideoconvert */
    if (!gst_element_link_many(source, capsfilter, vidconv, NULL)) {
        fprintf(stderr, "[Error] Failed to link source → capsfilter → vidconv\n");
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
