# QueueSense: what it is, how it works, and how it was tested

QueueSense is a local computer-vision app for one fixed camera, one queue, and one service counter. You give it a video. It finds people, follows them for that clip, and measures how long they wait. A small dashboard shows the annotated video, the queue counts, the visits, and a clearly labeled wait estimate.

It is a portfolio project for a BTech fresher. It does not train a model, recognize faces, or use a webcam. A tracker id is a temporary label for one pass through one video. It is not a person's identity.

## What was built

The program is a Python 3.11 app. The pieces are:

| Piece | What it does |
| --- | --- |
| `app.py` | Local Streamlit dashboard: upload a video, draw the two zones, start analysis, watch the result, download CSV |
| `src/queuesense/video.py` | Read the file, choose timestamps, write a playable annotated video |
| `src/queuesense/detector.py` | Load YOLO26n and run ByteTrack |
| `src/queuesense/zones.py` | Polygons, the foot point, and "service wins if a point is in both zones" |
| `src/queuesense/state_machine.py` | Decide when someone has joined, is waiting, is being served, has left, or was lost |
| `src/queuesense/prediction.py` | A simple one-server estimate, using only information already seen |
| `src/queuesense/storage.py` | SQLite database, one row per run |
| `src/queuesense/analytics.py` | Tables, the queue chart, CSV files |
| `src/queuesense/visualization.py` | Boxes, zone colors, and the two queue numbers drawn on the frame |
| `src/queuesense/evaluation.py` | Compare a saved run with a manual annotation file |
| `src/queuesense/cli.py` | Command line for processing and evaluation |

The pretrained model only supplies person boxes. Everything after that box is QueueSense code: zones, timing, visits, the estimate, storage, and the screen.

Default settings live in `configs/default.json`:

- Model: `yolo26n.pt` (the small COCO person detector from Ultralytics)
- Confidence: 0.25
- Device: `auto` (CPU unless this machine's PyTorch can see a GPU)
- Image size: 640
- Frame stride: 1, meaning every frame
- Join confirmation: 1.5 seconds
- Service confirmation: 1.0 second
- Exit confirmation: 2.0 seconds
- Missing-track grace: 3.0 seconds
- Estimate warmup: 3 completed services

## How a run works

Nothing is inferred when you move a slider. Analysis starts only when you press **Start analysis**, or when you run the command-line `process` command. Each run gets a new tracker, a new state machine, and a new id in the database.

```
video file
    -> probe (size, FPS, duration, clock)
    -> you draw queue and service polygons on the first frame
    -> for every frame, in order:
         detect people
         keep ByteTrack ids
         take the bottom-center of each box as the feet
         test that point against the polygons
         update the state machine using video time
         store the count and the estimate that could be made at that moment
         draw the frame
    -> write the annotated video, CSV files, and the SQLite row
```

### 1. The video clock

Durations use the video's own time, not how long the computer took to process the file. OpenCV's presentation timestamp is used when a short probe shows those values increasing. Otherwise each frame is dated `frame_index / FPS`. If the presentation clock later goes backwards, the reader switches to the frame-index clock and keeps time moving forward.

Processing speed is stored separately, as frames divided by wall-clock seconds. That number is how fast the analysis ran. It is not a waiting time.

Frame stride defaults to 1. If you set it to 2, every second frame is skipped, timestamps still refer to the source video, and the output file is written at half the frame rate so the clip lasts about the same time. Skipped frames are invisible to the tracker. A stride longer than the 3-second grace period can mark someone lost after one missed sample.

The annotated file is accepted only after OpenCV can decode a frame from it. On this PC the H.264 encoder is missing, so the writer tries it, logs an error, and falls through to a codec that works. If `ffmpeg` is installed, that file is then transcoded so the browser can play it.

### 2. The zones

You click the corners of the queue, then the corners of the service area. The points are stored as fractions of the frame width and height. A resized preview does not move the zone, because a click is divided by the width and height that came back with that click.

A zone needs at least three corners, a real area, and edges that do not cross. The foot point is the bottom-center of the box. If that point is inside both polygons, it counts as service, not queue. If the shop counter is not in the shot, the zone file can leave service out. Everyone outside the queue is then just outside, and service numbers are not claimed.

### 3. Detection and tracking

Ultralytics YOLO26n finds people. The person class is read from the model's own name list. On the COCO weights that id is 0, but the code looks up the word `person` instead of assuming 0.

ByteTrack, as shipped by Ultralytics, links boxes across frames into track ids. QueueSense passes `configs/bytetrack.yaml`. The track buffer is 72 frames, about three seconds at 24 fps, so a briefly hidden box can keep its id. That is still ByteTrack. There is no extra step that tries to glue two ids back into one person.

Track ids restart every run. The same person can be id 4 in one run and id 11 in the next. Inside one run an id can also jump when people overlap. That is recorded as a limitation, not hidden.

### 4. The state machine

Each track moves through explicit states. Time is video time, and a timer only advances on frames where that track was actually seen. A gap does not count as presence.

| State | Meaning |
| --- | --- |
| OUTSIDE | Not in the queue and not in service |
| JOIN_CANDIDATE | Seen in the queue, but not yet for 1.5 continuous seconds |
| WAITING | Join is confirmed. The join time is the start of that 1.5 seconds, not the later frame |
| SERVICE_CANDIDATE | Seen in the service zone, not yet for 1.0 second |
| IN_SERVICE | Service is confirmed. Service start is the beginning of that second |
| COMPLETED | They were in service, then seen outside it for 2 seconds |
| ABANDONED | They had joined, then were seen leaving the queue for 2 seconds without entering service |
| LOST | The track vanished for more than 3 seconds. This is not treated as "they left" or "they finished" |

Waiting time is service start minus join time. It exists only after service has been confirmed. Someone who walks through the queue for less than 1.5 seconds never becomes a visit. Someone who leaves without being served has an empty waiting time, not zero.

Service duration runs from the confirmed service start to the beginning of the 2-second exit, not to the later frame on which the exit was confirmed.

If the video ends while someone is still waiting or still being served, the visit is marked incomplete. The program does not invent a checkout.

A track that finishes can join again later. That second episode gets a new visit id. The track id and the visit id are different numbers.

Two counts are shown, because they answer different questions:

- **Observed queue:** how many foot points are inside the queue polygon on this frame.
- **Active waiting visits:** visits that are waiting or about to be confirmed in service, including tracks missing for less than 3 seconds.

A displayed 0 means the count was measured as zero. An em dash means there was nothing to measure, such as an average wait when nobody has finished.

### 5. The estimate

The estimate is labeled **ESTIMATE**. It is not a measured wait.

It needs three completed service durations from this same video. Until then the status is `insufficient_data` and no number is shown. The mean is the average of service durations completed so far in this run. For a new person arriving at this moment:

```
people already waiting × mean service time
+ max(mean service time − time the current customer has already been served, 0)
```

"People already waiting" is the active-waiting count, not the observed foot count. If two people are in service at once, the one-server assumption is false and the number is suppressed. Each estimate is stored on the frame that produced it, using only services that had already ended. Later frames are not used to rewrite it.

### 6. What gets saved

Each run is a folder under `outputs/` (or a folder you pass on the command line) plus a row in SQLite. The row stores the settings, the zone fractions, the model path, the person class ids, the device, and the processing speed. The folder contains:

- the annotated video
- `visits.csv` (one row per visit)
- `metrics.csv` (queue count and estimate at each processed time)
- `transitions.csv` (every state change)
- `summary.json` and `zones.json`

The dashboard can reopen a saved run without running the model again. CSV buttons download those tables.

## How you run it

From the project folder in PowerShell, with the virtual environment that was created for this machine:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open `http://localhost:8501`.

1. Upload a video, or load the synthetic rectangle clip. The rectangle clip only checks that a file can be read. It is not people.
2. Click the queue corners, then the service corners, and press **Save zones**.
3. Leave stride at 1 and confidence at 0.25 unless you have a reason to change them.
4. Press **Start analysis**. The first time, `yolo26n.pt` downloads into `models\` (about 5.3 MB). Later runs reuse that file.
5. Open the Results tab. Watch the annotated video, read the two queue numbers, and download the CSVs.

Command line, after the zones have been saved as JSON:

```powershell
python -m queuesense.cli process --video path\to\clip.mp4 --zones path\to\zones.json --device cpu --conf 0.25 --stride 1 --model models\yolo26n.pt
python -m queuesense.cli evaluate --db data\queuesense.db --run-id RUN_ID --ground-truth path\to\annotations.json
```

A fresh machine needs Python 3.11, then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` pins the CPU build of PyTorch, Ultralytics 8.4.202, OpenCV 4.14.0.94, Streamlit 1.66.1, and the other libraries that were actually installed here.

## How it was tested

There are two layers. They answer different questions.

### Automated tests

`python -m pytest` runs 55 tests. They do not download the model and they do not use real people. They feed the state machine fake timestamps and fake boxes.

They check:

- A point on the edge of a polygon counts as inside, and a self-crossing shape is rejected.
- A click on a resized preview converts to the same fraction as a click on the full frame.
- A passerby who is in the queue for less than 1.5 seconds never becomes a visit.
- A real join followed by service records join time and service start at the beginning of those intervals. Waiting time is the difference. Service ends at the start of the exit interval.
- Leaving the queue without service becomes abandoned, with an empty wait.
- Disappearing for less than 3 seconds and coming back keeps the same visit.
- Disappearing for more than 3 seconds becomes lost, not abandoned.
- A gap in the middle of the 1.5 second join does not let the unseen time finish the join.
- After a visit ends, the same track can open a new visit id.
- A visit still open at the last frame is incomplete. No fake checkout is written.
- The estimate stays blank until three services exist, then uses the formula above, and is suppressed when two people are in service.
- Two runs in one database do not mix their visits.
- A video file is accepted only if a frame can be decoded. An empty file is not treated as success.

The synthetic rectangle video only proves that reading and writing a file works.

### Real videos

On 5 November 2026, five free stock clips were downloaded and watched. Details, licenses, and SHA-256 hashes are in `evaluation/VIDEO_SOURCES.json`. The full test write-up is `evaluation/REPORT.md`.

Two clips were actually scored. Both were one continuous shot from a camera that did not move.

**Starbucks counter, 10.30 seconds, 720p.** A rope line and a register are visible. People are already waiting when the shot starts and still waiting when it ends. Manual count of feet inside the queue polygon: 5 people, sampled every 0.5 seconds (21 samples, 10 of them partly blocked by someone walking past).

- Queue-count mean absolute error: **3.81 people** on the first run, **3.76** after the debug rerun.
- The program often said 0, 1, or 2 while five people were still in the polygon.
- Three visits were opened and all three were marked lost by about 6.4 seconds, even though those people had not left. Track ids jumped into the hundreds. That is id switching under occlusion.
- The customer at the register was never counted as in service. The bottom of their box sat on a suitcase in the aisle, outside the service polygon.
- No completed wait existed, so waiting-time error was not computed.
- The estimate stayed `insufficient_data` for the whole file. That part behaved correctly.
- Analysis ran at 7.87 frames per second (31 seconds of processing). The annotated video is 10.302 seconds, matching the source.

**Sidewalk queue, 24.02 seconds, 1440p.** People wait outside a shop. The counter is not in frame, so no service zone was drawn and service numbers were not scored. The count used for comparison is 9, with every sample marked uncertain because people block each other and the nearest feet fall below the polygon. The true line is longer than the polygon.

- Queue-count mean absolute error: **2.29 people** over 24 one-second samples.
- The program's count stayed between 3 and 12 and averaged 6.96.
- 9 visits ended as lost and 4 were still incomplete when the file ended. Each visit had a different track id. None of these is a finished wait.
- Analysis ran at 6.87 frames per second (84 seconds). The annotated video is 24.024 seconds, matching the source.
- The estimate was again `insufficient_data`.

Three other downloads were watched and then excluded: a walking camera in an airport queue, a close-up of one customer with a cashier, and a shopping street where people are walking rather than waiting.

### What the tests do not prove

The 55 automated tests prove the timing rules and the file handling. They do not prove that YOLO finds people well.

The two real clips prove that detection and counting run on genuine fixed-camera footage, and they give a queue-count error for those short shots. They do not prove waiting-time accuracy or estimate accuracy. Both shots are too short, and the people are already in line at the first frame. A fair waiting-time test needs a still camera and several people who arrive, wait, and then get served.

## What you should expect when you use it

On a clear, still shot, boxes appear on people and the observed count moves when feet enter or leave the polygon you drew. Passersby who cross the queue for a moment should not become visits, because of the 1.5 second join rule. After someone has been standing in the zone for 1.5 seconds, a visit starts. If the detector keeps the same id until they reach the counter and then leave, you get a waiting time and, after three such services, an estimate.

On the airport Starbucks clip, that full story did not happen. People were found, the count was often low, ids broke when someone walked in front of the line, and visits were closed as lost. The estimate correctly refused to invent a number.

Draw the queue tightly around the standing line, and the service zone on the floor where the customer stands, not on the staff behind the counter. A zone that includes the walking aisle will count passersby. A zone that misses people's feet will count too few.

The useful interview points are the ones the code actually enforces: normalized zones, video time instead of the processing clock, confirmation that ignores gaps, separate track and visit ids, and an estimate that is stored only from the past and labeled as an estimate.
