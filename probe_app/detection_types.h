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
 *   int frameNumber = buf.getInt();
 *   int numDets     = buf.getInt();
 *   for (int i = 0; i < numDets; i++) { ... }
 *
 * Published to NT topic: /vision/detections
 */
typedef struct {
    uint32_t  frame_number;
    uint32_t  num_detections;
    Detection detections[MAX_DETECTIONS];
} DetectionFrame;
