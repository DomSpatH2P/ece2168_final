# Dominic Spatola ECE 2168 "AE"/TC with DBSCAN

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from SnnDbscan import SnnDbscan, get_clusters
import time
from math import atan, pi

# --- Parameters ---
NEUROMORPHIC = True #uses simulated SNN and center of mass if true, traditional DBSCAN if false
RESET_CENTROID_DISPLAY = True #whether to reset centroid display every frame
PIX_ON_THRESH  = 5
PIX_OFF_THRESH = -5
EPS            = 5
MIN_SAMPLES    = 3
MAX_MATCH_DIST = 10.0
EDGE_PAIR_DIST = 20.0   # max distance to consider ON/OFF edges same object
DISP_THRESH    = 1.0
CAMERA_PATH = 0
VIDEO_PATH = "DVD logo.mp4" #type filename or 0 or CAMERA_PATH for camera

# --- Camera setup ---
cam = cv2.VideoCapture(VIDEO_PATH) 
cam.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
centroids = np.zeros((48, 64), dtype=np.uint8)
traditional_mask = np.zeros((48, 64), dtype=np.uint8)
dbscan = SnnDbscan(48,64,EPS,MIN_SAMPLES)

ret, oldframe = cam.read()
oldframe = cv2.resize(oldframe, (320, 240)) #resize for videos
oldgray = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
# --- Helpers ---
def get_events(gray, oldgray): #simulate DVS; get difference in frames, downsample, and get thresholds
    diff = gray.astype(np.int16) - oldgray.astype(np.int16)
    diff = diff.reshape(48,5,64,5).mean(axis=(1,3))

    on_map  = (diff >  PIX_ON_THRESH).astype(np.uint8)
    off_map = (diff <  PIX_OFF_THRESH).astype(np.uint8)

    # Downsample each polarity separately
    on_ds  = on_map #.reshape(48, 5, 64, 5).max(axis=(1, 3))
    off_ds = off_map #.reshape(48, 5, 64, 5).max(axis=(1, 3))

    both_ds = on_ds + off_ds

    on_pts  = np.argwhere(on_ds > 0)
    off_pts = np.argwhere(off_ds > 0)

    # Display: ON=white, OFF=gray, nothing=black
    display = np.zeros((48, 64), dtype=np.uint8)+128
    display[off_ds == 1] = 0
    display[on_ds  == 1] = 255

    return (on_ds, off_ds, display) if NEUROMORPHIC else (on_pts, off_pts, display)


def get_centroids(points): #Traditional DBSCAN and labeling
    if len(points) < MIN_SAMPLES:
        return {}
    db     = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(points)
    labels = db.labels_
    for p in points[labels != -1]:
        traditional_mask[p[0],p[1]] = 255
    return [points[labels == cid].mean(axis=0) for cid in set(labels) if cid != -1]


def pair_edges(on_centroids, off_centroids): #attempts to pair on and off centroids from the same object by distance
    if not on_centroids or not off_centroids:
        return []

    on_pts  = np.array(on_centroids)
    off_pts = np.array(off_centroids)

    D = cdist(on_pts, off_pts)
    objects = []
    used_off = set()

    for i in range(len(on_pts)):
        j = D[i].argmin()
        if D[i, j] < EDGE_PAIR_DIST and j not in used_off:
            motion   = on_pts[i] - off_pts[j]   # trailing→leading
            midpoint = (on_pts[i] + off_pts[j]) / 2.0
            objects.append({
                "motion":    motion,
                "midpoint":  midpoint,
                "on_center": on_pts[i],
                "off_center": off_pts[j],
            })
            used_off.add(j)

    return objects

"""
def describe_motion(motion): #determine plane of motion; actual motion depends on whether background or obejct is darker
    if (motion[0] > DISP_THRESH) and (abs(motion[1]/motion[0]) < ):
        return "⇑⇓" #north/south
    if (motion[1] > DISP_THRESH) and (abs(motion[1]/motion[0]) <):
        return "⇐⇒" #east/west
    if (((motion[0] > DISP_THRESH) and (motion[1] > DISP_THRESH)) or ((motion[0] < -DISP_THRESH) and (motion[1] < -DISP_THRESH))):  
        return "⇖⇘" #northwest/southeast
    if ((motion[0] < -DISP_THRESH) and (motion[1] > DISP_THRESH)) or ((motion[0] > DISP_THRESH) and (motion[1] < -DISP_THRESH)):  
        return "⇙⇗" #northeast/southwest
    return None
"""

def describe_motion(motion): #determine plane of motion; actual motion depends on whether background or obejct is darker
    if (motion[0]*motion[0] + motion[1]*motion[1]) > DISP_THRESH*DISP_THRESH:
        if motion[1]==0: return "⇑⇓" #avoid divide by 0
        angle = atan(motion[0]/motion[1])
        if abs(angle) < pi/8: return "⇐⇒"
        if (angle >= pi/8) and (angle < 3*pi/8): return "⇖⇘"
        if (angle <= -pi/8) and (angle > -3*pi/8): return "⇙⇗"
        return "⇑⇓"
    return None

# --- Main loop ---
t = 0
t_average = 0 #average time 
average_centroids = 0 #average centroids
while True:
    start = time.perf_counter()
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.resize(frame, (320, 240))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    on_pts, off_pts, display = get_events(gray, oldgray)
    if(NEUROMORPHIC):
        on_core, on_border, on_b = dbscan.run(on_pts)
        on_mask = (on_core | on_border)
        on_centroids = get_clusters(on_mask,on_b,MIN_SAMPLES)
        off_core, off_border, off_b = dbscan.run(off_pts)
        off_mask = (off_core | off_border)
        off_centroids = get_clusters(off_mask,off_b,MIN_SAMPLES)
        cv2.imshow("neuromorphic mask",cv2.resize(127+(127*on_mask)-(127*off_mask),(320, 240), interpolation=cv2.INTER_NEAREST))
    else:
        traditional_mask.fill(0)
        on_centroids  = get_centroids(on_pts)
        off_centroids = get_centroids(off_pts)
        cv2.imshow("traditional mask",cv2.resize(traditional_mask,(320, 240), interpolation=cv2.INTER_NEAREST))
    objects       = pair_edges(on_centroids, off_centroids)

    if(RESET_CENTROID_DISPLAY): centroids.fill(128)

    num_centroids = len(objects)
    for idx, obj in enumerate(objects):
        direction = describe_motion(obj["motion"])
        if direction:
            print(f"t={t} object {idx}: {direction}"
                  f"  motion={np.round(obj['motion'], 1)}"
                  f"  at={np.round(obj['midpoint'], 1)}")
        on_center = obj['on_center']
        off_center = obj['off_center']
        centroids[int(on_center[0]),int(on_center[1])]=255
        centroids[int(off_center[0]),int(off_center[1])]=0

    # Display
    event_display = cv2.resize((display).astype(np.uint8),
                               (320, 240), interpolation=cv2.INTER_NEAREST)
    centroid_display = cv2.resize((centroids).astype(np.uint8),
                               (320, 240), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("centroids",centroid_display)
    cv2.imshow("events", event_display)
    cv2.imshow("frame",  gray)

    if cv2.waitKey(1) == ord('q'):
        break
    end = time.perf_counter()
    oldgray = gray
    t += 1
    t_average = t_average*(t-1)/t + (end-start)/t
    average_centroids = average_centroids*(t-1)/t + num_centroids/t 

cam.release()
cv2.destroyAllWindows()
print(f"Neuromorphic ({NEUROMORPHIC}); eps = {EPS}; average period {t_average} s; t_average; average centroids: {average_centroids}")