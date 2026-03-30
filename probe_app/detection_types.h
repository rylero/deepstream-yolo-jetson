#pragma once
#include <stdint.h>

#define MAX_DETECTIONS 64
#define FRAME_WIDTH    1280
#define FRAME_HEIGHT   720

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
 *   int    frameNumber  = buf.getInt();
 *   int    numDets      = buf.getInt();
 *   long   timestampUs  = buf.getLong();   // microseconds since Unix epoch
 *   float  fps          = buf.getFloat();
 *   int    _reserved    = buf.getInt();    // skip padding
 *   for (int i = 0; i < numDets; i++) { ... }
 *
 * Layout (offsets):
 *   0  frame_number    uint32
 *   4  num_detections  uint32
 *   8  timestamp_us    int64   (CLOCK_REALTIME, microseconds since Unix epoch)
 *  16  fps             float   (exponential moving average)
 *  20  _reserved       uint32  (padding)
 *  24  detections[]
 *
 * Published to NT topic: /vision/detections
 */
typedef struct {
    uint32_t  frame_number;
    uint32_t  num_detections;
    int64_t   timestamp_us;   /* microseconds since Unix epoch (CLOCK_REALTIME) */
    float     fps;            /* exponential moving average FPS */
    uint32_t  _reserved;      /* padding — keep detections[] 8-byte aligned */
    Detection detections[MAX_DETECTIONS];
} DetectionFrame;
