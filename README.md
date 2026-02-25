# QueueSense

QueueSense measures a single queue at a single service counter in a fixed-camera video. It is a local portfolio project for a BTech student preparing for a computer vision interview. A reviewer can upload one clip, draw the queue and the counter, and inspect waiting times, an explicitly labeled estimate, and the visits behind those numbers.

The intended user is the student and the interviewer sitting with them. The app runs on the student's machine. It does not train a model, recognize faces, or accept a live webcam.

The branch `demo/synthetic-history` is an educational demonstration. Its commit timestamps were set explicitly to dates in February 2026 (Asia/Kolkata). Those author and committer dates are synthetic. They are not a record of when the work was actually written.

## What the pretrained model does

Ultralytics YOLO26n, pretrained on COCO, supplies person bounding boxes. Ultralytics' ByteTrack implementation turns those boxes into track ids for the current pass. QueueSense code does everything after that: zone tests, the state machine, measured waits, the FIFO estimate, storage, evaluation, the annotated video, and the dashboard.

Track ids are association hypotheses inside one processing run. They are not names, faces, or permanent identities. An id switch is a known failure mode, and this version does not try to repair one.

## Architecture

```
app.py / cli.py
    pipeline.execute_run
        video.py          probe, timestamps, annotated writer
        detector.py       YOLO person class + ByteTrack
        zones.py          normalized polygons, ground point, overlap
        state_machine.py  per-track visit logic
        prediction.py     causal single-server estimate
        storage.py        SQLite
        analytics.py      tables and the queue chart
        visualization.py  boxes, zones, both queue numbers
```

Processing flow:

1. Probe the file: width, height, FPS, reported frame count, duration, and whether presentation timestamps increase.
2. Draw two polygons on the first frame. Store them as fractions of width and height.
3. Read frames in order. Default stride is 1, so every frame is processed.
4. Detect people and keep ByteTrack ids. The person class id is read from the model name map.
5. Use the bottom-center of each box as a stand-in for ground contact.
6. If that point is in both zones, the service zone wins.
7. Update the state machine using source-video time.
8. Store the visit, the transition, the queue counts, and the estimate that could be computed at that timestamp.
9. Write an annotated video, CSV files, and a SQLite run. A new run clears tracker memory and builds a new state machine.

Each run is a new directory under `outputs/` and a new row in `data/queuesense.db`. The row stores the settings, the resolved person class ids, the Ultralytics version, and the device.

## Setup on Windows PowerShell

Use Python 3.11. The versions below are the ones installed and tested in this workspace (Python 3.11.9).

```powershell
cd C:\Users\nikhi\OneDrive\Desktop\comp_vision
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest
streamlit run app.py
```

`requirements.txt` pins PyTorch's CPU wheels and points pip at `https://download.pytorch.org/whl/cpu`. The default device choice is `auto`: CUDA when `torch.cuda.is_available()` is true, otherwise CPU. Choose `cpu` or `cuda` in the dashboard or pass `--device` on the CLI. A CUDA request on a CPU-only install stops with an error instead of silently switching.

Open `http://localhost:8501`. The first analysis that uses the default model name `yolo26n.pt` downloads that weight file from Ultralytics into `models/` when the download lands in the working directory. Later runs reuse `models/yolo26n.pt`. The download needs a network connection and only happens when you start an analysis, not when the dashboard opens and not when the tests run.

Changing a slider does not run the model. Inference starts only from **Start analysis**.

## CLI

```powershell
python -m queuesense.cli process --video path\to\clip.mp4 --zones path\to\zones.json
python -m queuesense.cli process --video path\to\clip.mp4 --zones path\to\zones.json --device cpu --conf 0.25 --stride 1 --model yolo26n.pt
python -m queuesense.cli evaluate --db data\queuesense.db --run-id RUN_ID --ground-truth path\to\annotations.json
```

Optional flags on `process`: `--config`, `--model`, `--conf`, `--device`, `--stride`, `--db`, `--output-root`.

`evaluate` prints queue-count error, waiting-time error on matched visits, matched / missed / unmatched visit counts, prediction error only where a valid stored estimate exists, processing FPS, and the hardware recorded with the run. The report states that these scores compare the run with the annotation file. They are not a detector benchmark.

`data/sample_zones.json` and `data/sample_ground_truth.json` show the file formats. The ground-truth sample has empty lists on purpose. Fill it with your own annotations before treating a score as meaningful.

## Zones

In the dashboard, choose **Queue zone** or **Service zone**, then click the corners in order. **Undo point**, **Reset this zone**, **Reset both**, and **Save zones** are under the picture. A loaded JSON file uses the same normalized format:

```json
{
  "coordinate_space": "normalized",
  "queue": [[0.08, 0.20], [0.48, 0.20], [0.50, 0.92], [0.06, 0.92]],
  "service": [[0.55, 0.18], [0.92, 0.18], [0.94, 0.90], [0.58, 0.88]]
}
```

Coordinates are fractions of the source frame. The preview can be scaled for the window; the click is divided by the width and height reported with that click, so the stored polygon does not change with the preview size.

A zone needs at least three distinct points, a positive area, and edges that do not cross. On the video, the service polygon is tested first. Draw the service region on the counter and leave the standing area to the queue polygon. If every queue vertex falls inside the service polygon, the dashboard warns you: those points will never count as queue.

## State machine

States: `OUTSIDE`, `JOIN_CANDIDATE`, `WAITING`, `SERVICE_CANDIDATE`, `IN_SERVICE`, `COMPLETED`, `ABANDONED`, `LOST`. A visit that is still open when the file ends is marked `incomplete`. That end-of-file mark is not a completion and not an abandonment.

Default timers, all in source-video seconds:

| Timer | Default | Effect |
| --- | --- | --- |
| Join confirmation | 1.5 s | Continuous observed presence in the queue before a visit exists |
| Service confirmation | 1.0 s | Continuous observed presence in the service zone before service starts |
| Exit confirmation | 2.0 s | Continuous observed presence outside the relevant zone |
| Missing-track grace | 3.0 s | How long a confirmed visit stays open without a detection |

Join time is the start of the uninterrupted queue observations that were eventually confirmed, not the later frame on which the timer elapsed. Service start is the start of the confirmed service observations. Waiting time is service start minus join time. Service duration runs from service start to the start of the confirmed exit observations. A person who leaves the queue without entering service becomes `abandoned` after the exit timer. A person who disappears for longer than the grace period becomes `LOST`. Time during which the track was absent does not count toward join, service, or exit.

The same track id can open a later visit with a new visit id. A passerby who leaves before 1.5 seconds never becomes a visit.

Two counts are shown on the video and in the chart:

- **Observed queue count**: ground points inside the queue zone on that frame.
- **Active waiting visits**: visits in `WAITING` or `SERVICE_CANDIDATE`, including tracks that are missing inside the grace period and people still inside the exit-confirmation window.

A displayed 0 is a real measurement. An em dash or the word "unavailable" means the value was never measured.

## Timestamps and frame stride

Event durations use source-video time. OpenCV's `CAP_PROP_POS_MSEC` is used when a short probe shows those values increasing. Otherwise each frame is dated `frame_index / FPS`. If a presentation timestamp later stops increasing, the reader switches to the frame-index clock and keeps the series monotonic.

Wall-clock time is stored only as processing speed (frames processed divided by analysis time). It is not a wait.

Frame stride defaults to 1. Stride N keeps the source timestamp of every Nth frame and writes the annotated file at `source_fps / N`, so playback duration stays aligned with the source. Frames that are skipped are invisible to the detector and the state machine. A stride longer than the grace period can mark a track `LOST` after one missed sample, and a confirmation shorter than the gap between samples can be missed.

The annotated file is accepted only after OpenCV decodes a frame from it. If no codec can do that, the run fails and the message names the codecs that were tried. When `ffmpeg` is on `PATH`, a non-H.264 file is transcoded to H.264 so the browser player has a better chance. A missing `ffmpeg` leaves the OpenCV file in place; the dashboard says so if the player may stay blank.

## Waiting time and the estimate

Measured waiting time exists only for a visit that reached confirmed service: service start minus join time. Abandoned, lost, and incomplete visits that never started service have an empty wait, not a zero.

The estimate is separate and labeled **ESTIMATE**. It needs three completed service durations from this run. Until then the status is `insufficient_data`. The mean is the running mean of completed service durations observed so far (an expanding window). For a hypothetical new arrival at that timestamp:

```
active waiting visits × mean service
+ max(mean service − elapsed service of the person in service, 0)
```

Elapsed service is video time since the backdated service start. If two people are in service, the single-server assumption is false and the estimate is suppressed (`one_server_violation`) instead of shown as a number. Predictions are written on the metrics row for the frame that produced them, using only samples completed by that frame.

Assumptions, also written in the dashboard: one server, first-come service, future service times similar to the services already completed in this clip, and the active-waiting count as the queue ahead of a new arrival. That count is not the observed occupancy.

## Dashboard

Setup: upload or the synthetic rectangle clip, metadata, zone editor, model path, confidence, device, image size, stride, and the four timers. Results: saved runs, annotated video, both queue numbers, the time chart, mean and median completed waits, outcome counts, service-duration stats, the estimate, the visit table, and CSV downloads for visits, metrics, and transitions.

The synthetic clip is a moving rectangle for the player and the file probe. It does not contain people. A detector run on it does not measure detection accuracy.

## Recording a short demonstration

Use a tripod or a phone propped so the camera does not move. Frame one queue and one counter. Ask the people in view for consent, and tell them the software does not try to recognize them. Record 30 to 60 seconds at 720p, with the queue entering from one side and the counter visible. Keep the camera still, avoid strong backlight, and leave a visible gap between the standing area and the counter so the two polygons are easy to draw. Export the clip without relying on a player-only rotation flag; OpenCV often ignores rotation metadata.

## 90-second demonstration script

- 0–10 s. One fixed camera, one queue, one counter. The model finds person boxes. The rest of the measurement is ours.
- 10–25 s. Upload the consented clip. Show FPS, size, and duration. Click the queue polygon, then the service polygon, and save.
- 25–40 s. Leave stride at 1 and confidence at 0.25. Press **Start analysis**. Point out that the page does not re-run the model when other widgets change.
- 40–60 s. Play the annotated video. Read the two labels: observed queue, and active waiting visits. They answer different questions.
- 60–75 s. Open the visit table. Mean wait is blank until a visit completes service. The estimate stays on "insufficient data" until three services complete, and it is labeled as an estimate.
- 75–90 s. Download the visits CSV. Say that track ids can switch, the camera must stay fixed, and the estimate assumes one server and FIFO.

## Limitations

- Occlusion and close standing make ByteTrack swap or drop ids. A swapped id splits one person's wait or glues two people together. This version does not reconnect them.
- The ground point is the bottom-center of the box. Perspective, a high camera, and feet hidden by the counter move that point relative to the floor.
- Low light, backlight, and motion blur change the detector. Confidence is a single threshold for the boxes ByteTrack is allowed to see.
- A moving camera breaks the zones. The polygons are in image coordinates, not a floor plan.
- The estimate assumes one server and first-come service. It is suppressed when two people are confirmed in service. It still uses a simple remaining-time approximation and the active-waiting count, which includes brief gaps.
- A single missed detection resets a confirmation timer. The grace period keeps an already confirmed visit open; it does not let unobserved time finish a join, a service entry, or an exit.
- Phone videos with rotation metadata can appear sideways.

## Dependencies and licenses

Tested in this session with Python 3.11.9 on Windows:

| Package | Version | Role | License |
| --- | --- | --- | --- |
| torch / torchvision | 2.15.0+cpu / 0.29.2+cpu | runtime for Ultralytics, CPU build | BSD-style PyTorch license |
| ultralytics | 8.4.202 | YOLO26n and ByteTrack | AGPL-3.0, or a commercial Ultralytics license |
| lap | 0.5.14 | assignment step inside Ultralytics ByteTrack | BSD-2-Clause |
| opencv-python | 4.14.0.94 | video, geometry, drawing | Apache-2.0 |
| streamlit | 1.66.1 | local dashboard | Apache-2.0 |
| streamlit-image-coordinates | 0.4.1 | click capture for zones | MIT |
| pandas | 3.1.1 | tables and CSV | BSD-3-Clause |
| plotly | 7.1.0 | queue chart | MIT |
| pytest | 9.2.0 | tests | MIT |

OpenCV 5 exists. This project stays on OpenCV 4.14 because Ultralytics 8.4.202 declares `opencv-python>=4.7` and the 4.x calls used here (`VideoCapture`, `VideoWriter`, `pointPolygonTest`) are the ones exercised by the tests. YOLO26 is the current Ultralytics detection family; `yolo26n.pt` is the small COCO checkpoint. The default tracker name in Ultralytics is no longer ByteTrack, so QueueSense passes `tracker="bytetrack.yaml"` explicitly. The person class is resolved by the label `person` in `model.names`.

QueueSense's own code is MIT (`LICENSE`). Ultralytics and the YOLO weights are AGPL-3.0 unless you have an Ultralytics enterprise license. Running the project locally for study is the intended use. Shipping a combined application can create AGPL obligations. This note is not legal advice.

ByteTrack is the algorithm from Zhang, Sun, Jiang, Yu, Weng, Yuan, Luo, Liu, and Wang, ECCV 2022. The code executed here is Ultralytics' `BYTETracker`. The COCO dataset is the source of the pretrained boxes; this repository does not redistribute COCO images.

## Tests and verification

Tests build visits from synthetic timestamps and boxes. They do not download weights. `tests/test_video.py` writes a tiny moving-rectangle clip to prove that video reading, stride, and the codec check work. Those rectangles are not people, and a passing test is not a detection score.

```powershell
python -m pytest
```

Verification recorded for this workspace on Python 3.11.9, Windows, CPU-only PyTorch 2.15.0+cpu (`torch.cuda.is_available()` is false):

```text
.\.venv\Scripts\python.exe -m pytest
53 passed in 16.43s
```

The tests use synthetic timestamps, boxes, and a moving-rectangle video. They passed for polygon boundaries and normalized clicks, a passerby, a confirmed join and service, abandonment, a brief gap, a loss past the grace period, missing frames during confirmation, a rejoin with a new visit id, incomplete visits at the end of a clip, variable timestamp spacing, service-over-queue priority, estimate warmup and the FIFO formula, SQLite isolation between runs, video read/write, and a codec that claims success without a decodable frame.

`python -m queuesense.cli --help` lists `process` and `evaluate`. `streamlit run app.py` came up at `http://localhost:8501`. In the browser, the synthetic rectangle clip showed 320×240, 30 frames, 10 FPS, and 3.0 seconds. The zone editor mounted, **Start analysis** stayed disabled until two polygons are saved, and saving with no points reported that the queue zone needs at least 3 points. Changing widgets did not load the model.

This PC has no OpenH264 DLL, so the `avc1` and `H264` writers log an error and the code moves on. The rectangle file was still readable, and the probe selected presentation timestamps.

`yolo26n.pt` (5.3 MB) downloaded from the Ultralytics v8.4.0 release into `models/yolo26n.pt`. The loaded name map resolved person to class id 0. Three frames of the rectangle clip returned 0 person boxes, and `reset_tracker()` completed. That checks the tracking API. It does not measure detection quality.

A later session downloaded permitted stock clips and wrote the results in [evaluation/REPORT.md](evaluation/REPORT.md). On the fixed-camera Starbucks line, queue-count MAE was 3.8 people over 21 manual samples, and the clip was too short for a completed wait. The sidewalk queue, scored only for counting, had MAE 2.3 people over 24 ambiguous samples. Waiting-time prediction stayed on insufficient data in both runs.

## Interview talking points

1. Zones are stored as fractions of the frame, and a click is converted with the width and height that came back with the click. Resizing the preview cannot silently move the queue.
2. Confirmation uses source-video time and only advances across frames where that track was actually observed. A gap inside the grace period keeps a confirmed visit alive, and the same gap does not finish a join, a service entry, or an exit.
3. Track ids and visit ids are different. A track can finish and later open visit 2. Nothing in this version treats the id as a person, and an id switch is left as a limitation instead of hidden behind a recovery heuristic.
4. The estimate is stored on the frame that produced it, from services that had already ended. Three samples are required. Two people in service suppress the number. Measured waits and the estimate are separate fields.
5. Every run gets a new state machine, a tracker reset, and its own SQLite id. The row stores the settings and the person class ids taken from that model's names, so a later evaluation does not assume that "person" is class 0.
