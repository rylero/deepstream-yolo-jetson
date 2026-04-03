#pragma once
#include <stdint.h>

#define MAX_DETECTIONS 100
#define FRAME_WIDTH    1280
#define FRAME_HEIGHT   800

/*
 * Single object detection result.
 * Bounding box coordinates are normalized to [0.0, 1.0].
 */
typedef struct {
    int32_t class_id;
    float   confidence;
    float   left;    /* x of top-left corner, normalized */
    float   top;     /* y of top-left corner, normalized */
    float   width;   /* normalized width  */
    float   height;  /* normalized height */
    char    label[32];
} Detection;

/*
 * One frame's worth of detections, packed for NetworkTables NT_RAW transport.
 *
 * Robot-side unpacking (Java example):
 *   ByteBuffer buf = ByteBuffer.wrap(rawBytes).order(ByteOrder.nativeOrder());
 *   int    frameNumber  = buf.getInt();    //  0
 *   int    numDets      = buf.getInt();    //  4
 *   long   timestampUs  = buf.getLong();   //  8  microseconds since Unix epoch
 *   float  fps          = buf.getFloat();  // 16
 *   int    sourceId     = buf.getInt();    // 20  camera index (0, 1, ...)
 *   for (int i = 0; i < numDets; i++) { ... }
 *
 * Published to NT topics: /vision/detections/0, /vision/detections/1, ...
 */
typedef struct {
    uint32_t  frame_number;
    uint32_t  num_detections;
    int64_t   timestamp_us;   /* microseconds since Unix epoch (CLOCK_REALTIME) */
    float     fps;            /* exponential moving average FPS */
    uint32_t  source_id;      /* camera index — keeps detections[] 8-byte aligned */
    Detection detections[MAX_DETECTIONS];
} DetectionFrame;
