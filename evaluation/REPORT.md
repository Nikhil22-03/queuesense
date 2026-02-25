# QueueSense real-video test, 5 November 2026

This report covers footage that was actually downloaded and watched. Synthetic tests are separate and are not used as accuracy numbers here.

## What was validated

| Capability | Result |
| --- | --- |
| Person detection and queue counting on a fixed camera | Tested on two clips. Queue-count error is reported below. |
| Tracking and queue state | Observed. Track ids break up. No completed visit was produced. |
| Completed waiting time | Not tested. Neither clip shows a person join the queue and then reach service. |
| Wait-time prediction | Not scored. The app correctly stored `insufficient_data` because no service durations were completed. |
| Output timing | Checked. Annotated files match the source duration and frame count. |

## Footage

Permissions and hashes are in [VIDEO_SOURCES.json](VIDEO_SOURCES.json). Raw files stay in `data/external_videos/` and are gitignored.

Five candidates were downloaded on 5 November 2026:

1. **Starbucks, airport** (Pexels 37130586), 10.30 s, 1280×720, 23.976 fps. Fixed camera, one continuous shot, a rope line and a visible register. Used as the development clip. Suitable for counting and a limited tracking look. Not suitable for a finished waiting time: the people are already in line at 0 s and still there at 10 s.
2. **Sidewalk queue** (Pexels 10467868), 24.02 s, 2732×1440, 23.976 fps. Fixed camera, one continuous shot, people waiting outside a shop. The shop counter and staff are not visible, so this clip was scored for counting only. The page tags include "timelapse", but frames at 0 s, 12 s, and 23.7 s show ordinary walking speed and a locked background, so the file was not treated as sped-up footage.
3. **Airport terminal** (Pexels 29576550), 12.89 s. Excluded. The camera walks into the line; a passport covers the lens at 6.4 s and the end frame is blurred.
4. **Cashier close-up** (Pexels 8421362, Kampus Production), 11.96 s. Excluded. One customer is already at the counter with a shopkeeper. There is no waiting line.
5. **Frankfurt shopping street** (Pixabay 126503, AchimWeidner), 29.57 s. Excluded. Fixed camera, but people are walking past stores, not waiting for service.

The Pexels page widget showed durations such as 30:10 for the Starbucks clip. OpenCV's frame count and FPS are the numbers above.

## How the runs were made

Hardware: Windows, Python 3.11.9, torch 2.15.0+cpu, CUDA not available, opencv 4.14.0, ultralytics 8.4.202, YOLO26n. Person class id from the model names: `[0]`. Confidence 0.25, image size 640, stride 1, device cpu.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m queuesense.cli process --video data\external_videos\pexels_37130586_starbucks_720p.mp4 --zones evaluation\starbucks\zones.json --device cpu --conf 0.25 --stride 1 --model models\yolo26n.pt --db evaluation\starbucks\queuesense.db --output-root evaluation\starbucks\runs
.\.venv\Scripts\python.exe -m queuesense.cli process --video data\external_videos\pexels_37130586_starbucks_720p.mp4 --zones evaluation\starbucks\zones.json --device cpu --conf 0.25 --stride 1 --model models\yolo26n.pt --db evaluation\starbucks\queuesense.db --output-root evaluation\starbucks\runs_tuned
.\.venv\Scripts\python.exe -m queuesense.cli process --video data\external_videos\pexels_10467868_street.mp4 --zones evaluation\street\zones.json --device cpu --conf 0.25 --stride 1 --model models\yolo26n.pt --db evaluation\street\queuesense.db --output-root evaluation\street\runs
```

The second Starbucks command is the post-debug rerun. It uses the widened service polygon and `configs/bytetrack.yaml`. The street run uses that same frozen tracker file and confidence. Its zone is different because the camera is different, and it has no service polygon.

Zones:

- [evaluation/starbucks/zones.json](starbucks/zones.json) and [evaluation/starbucks/zone_preview.jpg](starbucks/zone_preview.jpg)
- [evaluation/street/zones.json](street/zones.json) and [evaluation/street/zone_preview.jpg](street/zone_preview.jpg)

Polygons are normalized to the source frame. Previews were drawn with those fractions times the real width and height, then checked against the picture.

## Ground truth

Counts are people whose feet are inside the queue polygon. Passersby in the aisle are not counted. Staff behind the Starbucks counter are not counted. Ambiguous frames are still given a count, and the note says why.

Starbucks, 21 samples every 0.5 s from 0.0 through 10.0: count **5**. Ten of those samples are marked ambiguous because a passerby overlaps the line (2.5, 3.0, 4.0, 5.5, 6.5, 8.0, 8.5, 9.0, 9.5, 10.0 s). A partly hidden person behind the hat could make the count 6. File: [evaluation/starbucks/ground_truth.json](starbucks/ground_truth.json).

Five people were already in that polygon at 0 s and still there at the end (`gt-q1` through `gt-q5`). The customer in the vest was already at the register (`gt-s1`). None of these are given a waiting time, because the join and the service start were not both observed.

Street, 24 samples every 1 s from 0 through 23: count **9**, and every sample is ambiguous. The social line is longer than 9. The nearest people's feet fall below the polygon, so they are outside the region the program measures. Overlap makes an exact count uncertain by about 3 people. No service visit was annotated. File: [evaluation/street/ground_truth.json](street/ground_truth.json).

Predicted visits were not matched to those incomplete people. There is no complete ground-truth visit to match. The predicted visits are therefore reported as extra.

## Measured results

### Starbucks baseline, before the debug edits

Run `20261105-153317-1c9a5027`.

- Queue-count MAE: **3.810 people** over 21 timestamps (10 ambiguous).
- Waiting-time MAE: not scored. 0 complete ground-truth visits.
- Predicted visits: 3, all `lost`. Matched complete visits: 0. Missed complete visits: 0. Extra predicted visits: 3.
- Incomplete ground-truth observations: 6, unscored.
- Processing: 7.87 fps, 31.37 s of analysis time, 247 frames.
- Annotated file: 247 frames, 23.976 fps, duration 10.302 s. Source duration 10.302 s.
- Prediction: `insufficient_data` on every stored row. That is the correct status. No prediction error was computed.

At 0.00 s the overlay reads "Observed queue: 6" while five people are clearly standing in the polygon. At 3.0 s it reads "Observed queue: 1" and "Active waiting visits: 3" while the same line is still there; track 4 is `WAITING visit-0001`, and the register customer (track 2) is `OUTSIDE` because the bottom-center of the box sits on the suitcase in the aisle, outside the service polygon. At 9.5 s a passerby covers the line and the overlay reads "Observed queue: 0". By 6.38 s all three confirmed visits have become `LOST`. Track ids in the log jump into the 60s and 100s within ten seconds, which is an id switch, not three people leaving.

### Starbucks after the debug edits

Run `20261105-153646-6d997f08`. This is not an untouched holdout. The service polygon was widened, status-line spacing was fixed, and ByteTrack `track_buffer` was raised from 30 to 72 frames.

- Queue-count MAE: **3.762 people** (21 samples). Essentially unchanged.
- Visits: still 3 `lost`, 0 completed. The register customer was still not inside the service polygon.
- Processing: 8.44 fps, 29.25 s.
- The status text at 3.0 s now reads "Active waiting visits" in full. On the baseline frame the second line collided with the first and the word was cut.

A longer tracker buffer did not keep the queue ids alive through the passerby. That failure remains.

### Street queue, frozen settings, no service zone

Run `20261105-154132-0ee46fbe`.

- Queue-count MAE: **2.292 people** over 24 timestamps. All 24 are ambiguous. The system count stays between 3 and 12 and averages 6.96, against a manual estimate of 9.
- Waiting time: not scored. Service zone: not used.
- Predicted visits with a join time: 13 (9 `lost`, 4 still `incomplete` at the end of the file). Matched complete visits: 0. Extra relative to complete ground truth: 13. Those 13 are split tracks of people who were already waiting, not 13 extra customers.
- Processing: 6.87 fps, 83.84 s, 576 frames.
- Annotated file: 576 frames, 23.976 fps, duration 24.024 s. Source duration 24.024 s.
- Prediction: `insufficient_data`.

At 0 s and 12 s the overlay shows separate boxes on the woman in white, the grey hoodie, and several people further along the line, with visit ids such as `visit-0001` on track 4. Boxes also cover the stroller and miss the dog, which is correct. Several people at the near edge are labeled `OUTSIDE` because their ground point falls below the polygon. That is the main reason the count is below the length of the social line.

## Code changes from these runs

- Status lines on the annotated frame are spaced from the measured text height, so one line no longer covers the next. Test: `tests/test_visualization.py`.
- A zone file may omit `service` when no counter is visible. The street run uses that. Test: `tests/test_zones.py`.
- `configs/bytetrack.yaml` sets `track_buffer` to 72. The Starbucks rerun loaded that file. The visit outcome did not improve.

`python -m pytest -q` after these edits: **55 passed**.

## What remains untested

Completed waiting time and the numeric wait estimate need a fixed camera and several people who join a queue and then get served. These clips are 10 s and 24 s, and the people are already waiting when the shot starts. Short footage and the small, partly ambiguous counts mean the MAE values are a check of this setup, not a general accuracy claim.
