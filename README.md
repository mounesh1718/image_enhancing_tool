# Image Quality Enhancer

An offline desktop app that improves the images inside **PDF** and **Word (.docx)** documents, or standalone image files. It keeps each image's displayed size, position, cropping and orientation, and leaves the rest of the document unchanged. You review every change, and only the approved ones are written to a **new** file. The original is never modified.

## Run it

Requirements:

- .NET 10 SDK
- Python 3.11 or later on `PATH`, or set `IQE_PYTHON` to the interpreter
- The worker's Python packages:

  ```
  pip install -r worker/requirements.txt     # numpy, opencv-python(-headless), PyMuPDF
  ```

Start the app:

```
dotnet run
```

Run the tests:

```
dotnet test ImageQualityEnhancer.slnx           # C#: services, DOCX, orchestration, export, headless UI
python -m unittest discover -s worker/tests     # Python worker: image I/O, enhancement, PDF
```

To save UI screenshots during the tests, set `IQE_SCREENSHOT_DIR` before running them.

## How it works

```
Avalonia UI (MVVM)  ->  JobOrchestrator (C#, background)  ->  Python worker (JSON lines over stdin/stdout)
                         |- DOCX: Open XML SDK (read) + streaming zip rewrite (export)
                         |- queue saved to Temp/Job_<id>/metadata/queue.json after every batch
                         '- workspace: original/ extracted/ enhanced/ thumbnails/ metadata/ export/
```

A job goes through these stages:

1. **Copy.** The document is copied into the job workspace.
2. **Scan.** Every image is listed with its exact placement. PDF scanning works in page chunks.
3. **Extract.** Each unique image is extracted once.
4. **Enhance.** Images are processed in batches sized by memory.
5. **Review.** You approve or reject each enhancement.
6. **Export.** Approved images are replaced, the result is validated, and only then is it moved to the location you chose.

Key source files:

| Area | Files |
|---|---|
| Queue, batches, pause/resume/cancel, crash recovery | `Services/JobOrchestrator.cs`, `BatchPlanner.cs`, `PauseGate.cs` |
| Export, validation, diagnostics | `Services/JobOrchestrator.Export.cs` |
| DOCX scan / export / validate | `Services/Docx/` |
| Worker protocol and process | `worker/protocol.py`, `Services/PythonWorkerClient.cs` |
| Image analysis and enhancement | `worker/analyse.py`, `worker/enhance.py`, `worker/image_io.py` |
| PDF | `worker/pdf_ops.py`, `worker/pdf_content.py` |
| UI | `Views/`, `ViewModels/` |

## Safety guarantees (each covered by tests)

- **Same pixel size.** Every enhanced image has exactly the width and height of its original. The worker refuses to return anything else.
- **The original is never written.** Exports go to a new file, and saving over the original is refused.
- **Outputs are validated before they are published.**
  - In both formats, every untouched part must stay byte-identical: PDF objects and DOCX zip entries, text, fonts, vector graphics and unapproved images.
  - **PDF only:** pages must render.
  - **DOCX only:** every image keeps its displayed size, crop and position, and Word's schema validator must report no new errors.
- **Interrupted runs never leave a half-written file.** Outputs are written to a temporary name and renamed only once complete.
- **One bad image never stops a job.** It is marked failed and the original image is kept. A worker crash triggers a restart and one retry of that image.
- **Enhancement is conservative.**
  - Already-good images are left unchanged.
  - **Clean digital pictures keep their look at every level.** Charts, tables, flowcharts, diagrams and drawings saved from software (no grain, a background of one exact colour, dark ink) have nothing unwanted to remove. Their colours, shading (gradients), grey boxes, lines and text are kept exactly. Only measured blur is reversed, and nothing is whitened, darkened, thickened or recoloured. In a real 91-picture document, all clean pictures were kept unchanged at Level 4.
  - **Charts on scanned paper keep their bars, colours and grid lines.** Paper cleaning recognises filled areas (bars, boxes, bands) by their sharp edge and never turns them white. Long thin lines (grid lines, axes, table rules) are never smoothed or removed as specks, and coloured areas keep their colour.
  - Text, diagrams and chemical structures get only light processing, never local contrast changes, and their colours are never altered.
  - Fine texture is not mistaken for noise.
  - Colour grain from scans and cameras is not mistaken for colour, so grainy grey paper turns evenly white instead of into dots. Very grainy photos get no local contrast boost, which would bring the grain back.
  - Results with large changes, possible artefacts or text/diagram content need individual review; "Approve safe ones" skips them.
- **PDF image storage formats.** JPEG, JPEG 2000, Flate, LZW, RunLength and raw pixels are enhanced, including when wrapped in ASCII85 or ASCIIHex text encoding, as ReportLab and other PDF writers store images. Unused objects in the PDF are kept, so untouched parts stay byte-identical.
- **Kept unchanged, with a reason shown:**
  - vector graphics (SVG/EMF/WMF and PDF vector drawings)
  - inline PDF images
  - CMYK, palette and spot-colour PDF images
  - 16-bit and black-and-white images
  - OLE previews (e.g. ChemDraw), whose program redraws them
  - SVG fallback PNGs
  - linked images
  - animated images
  - images over the size or memory limit
- **Passwords are never saved.** Encrypted PDFs stay encrypted.
- **AI never invents text.** AI-sharpened text is kept only when every digit it shows was readable in the original (see [AI repair](#ai-repair)).
- **AI never runs or approves by itself.** It runs only when you click **Try AI enhancement** or **Try AI on eligible images**. A usable AI version is applied only to pictures you have not decided yet, the OpenCV result is kept so you can switch back, nothing is approved, and changing the version of an approved picture asks first.
- **A redrawn structure is never approved automatically.** It must read back as the identical molecule, and a person must approve it.
- **Signed PDFs need confirmation.** The app warns that changing images invalidates the signature in the new copy, and nothing is written until you confirm.
- **Temporary files stay local and are removed** when you finish or cancel a job. Diagnostics contain no file names or document content.

## Measured limits

See [Phase 8 results](#phase-8-results) below. These are measurements on one machine, not guarantees.

## Using the app

- **Start page, two steps.**
  1. *Choose how to enhance:* strength 1–4 and chemical structures (*Improve only*, *Redraw when verified* or *Redraw all*). Bolder text and the text check are under **More options**. AI is not chosen here: it is offered on the review screen, after the normal processing (see [Try AI after processing](#try-ai-after-processing)).
  2. *Add files to start:* drop files or click **Choose files…**.
- Unfinished jobs are listed at the top under **Continue where you left off**.
- **Review screen.**
  - The list on the left shows each image with its status and your decision. The first image to decide opens by itself.
  - **Approve** or **Keep original** sits right under the picture. After a decision, the next image to decide opens.
  - The review screen shows one short instruction beside **Approve** and **Keep original**. Photos and ordinary images do not show chemical checks. **Try another strength** re-processes the shown image at level 1–4. **Processing details** (collapsed initially) lists the measurements, steps, text check and notes. **Fine-tune (advanced)** has manual sliders.
  - **AI enhancement (optional)** under the picture says whether AI may help it, for example *Grain detected — SCUNet may help.* or *No suitable AI repair detected.* **Try AI enhancement** tries it on the shown picture; **Try AI on eligible images (N)** at the top tries it on every picture where it may help. See [Try AI after processing](#try-ai-after-processing).
  - **Approve safe ones** approves every result that needs no careful look. **Keep all originals** rejects every change.
  - **Save new copy…** writes the new document. **Done** then removes the temporary files.
  - **‹ Back** keeps the job for later. **⋯** holds *Discard this job* and the technical details.

## Reviewing results

- **Groups.** Every picture is sorted into one of four groups, shown above the image list with their counts: **Chemical diagrams**, **Flowcharts and graphs**, **Tables** and **Other images** (photos, logos, screenshots, maps, ...). Click a group to list only its pictures; it combines with the status filter (*To decide*, *Approved*, ...). A sheet of several figures is counted in each group it contains. See [Picture groups](#picture-groups-classification).
- **Doubtful pictures ask for their group; confident ones do not.** When the classifier is sure, its group is simply used (each row shows it, for example *Table · 99 %*, and **Processing details** shows the full suggestion). Only a doubtful picture shows a question with a group chooser: the classifier is unsure (below 60 %, for the picture or one of its panels), it looks chemical but could not be read reliably, it could not be classified, or you already changed it. **Use the suggestion** goes back. Your choice applies to every place the picture appears and is saved with the job.
- **Plain-language explanations.** Every reason a picture was kept or could not be processed, every point to check before approving, and every chemistry check is written in plain words with what to do, for example *Already clear. Nothing needed fixing, so the original is kept exactly as it is.* or *The picture's data is damaged and could not be opened. The original stays in the document unchanged.*
- **History on demand.** Jobs saved for later are not listed on the start page: the **History** button (clock, top right, with the number of saved jobs) opens them in a side panel to continue or remove. Likewise, the levels tried on a picture are behind the clock button next to the strength buttons; only a warning stays on the page when a lower level kept more of the original.

- **Compare modes:** side by side, or a before/after slider (drag the divider).
- **Zoom:**
  - The mouse wheel zooms around the cursor, and dragging moves the image.
  - Double-click switches between Fit and 100 %.
  - Side-by-side views zoom and move together.
  - Previews load at full resolution (up to 4096 px wide), so zooming shows real pixels.
- **Full width** hides the image list so the comparison uses the whole window.
- **Drag and drop:** drop a PDF, Word document or image files onto the start page.
- **Keyboard shortcuts:**

  | Key | Action |
  |---|---|
  | `+` / `−` | Zoom in / out |
  | `0` | Fit |
  | `1` | 100 % |
  | `A` | Approve |
  | `K` | Keep original |
  | `S` | Slider on/off |
  | `E` | Full width on/off |
  | `↑` / `↓` | Previous / next image |

## Report

After processing, **Save report…** on the review screen saves a self-contained HTML file (thumbnails embedded, no internet needed). It lists:

- **Summary.** Images found, re-altered (enhanced), kept unchanged, failed and approved. The redrawn count is always 0, because the tool improves quality only and never redraws. When AI was tried: how many pictures, how many AI versions were usable or not, and how many you chose.
- **For each image:**
  - where it is (page, body, header, table)
  - its type (photo or text/diagram)
  - before/after thumbnails
  - the reason, if it was kept unchanged
  - every step applied, explained in plain words
  - when AI was tried on it: **Version chosen** (*the AI version (SCUNet)* or *the OpenCV result*) and what the AI checks found; the steps listed are those of the chosen version
- **Quality before → after:** sharpness, noise, blur and tonal range.
- **Accuracy against the original:**
  - **Edges kept:** lines, letters and shapes still present.
  - **New edges:** possible artefacts.
  - **Structural similarity:** SSIM.
  - **Brightness change.**
- **Rating:**
  - **High:** at least 95 % of edges kept, at most 15 % new edges, and nothing flagged.
  - **Good:** at least 85 % of edges kept and at most 30 % new edges.
  - **Check:** anything else. Look at it zoomed in before approving.

## Composite images (sheets of panels)

An image that is a sheet of framed panels (a figure sheet, a page of boxed images) is processed **panel by panel**: each panel is found and treated as an image of its own, so a chemical structure is redrawn or cleaned, a photo gets photo treatment (or AI clarity), and a table or paragraph gets a white background and crisp text. Titles, gutters and frames are kept exactly as they are. The review screen and the report list what happened to each panel.

**Speck removal.** On cleaned paper, isolated dark specks (photocopy and scanner dirt) far from any text or line are removed and filled from the surrounding paper. Dots near letters (full stops, decimal points, the dot of an i) and short line-shaped marks (bonds, dashes, minus signs) are always kept.

**Measured on a 9-panel test sheet** (noisy and blurred structures, a micrograph, a photo, a table, a chart, a circuit and a paragraph): every text panel came out with a white background and sharp text, the table and chart kept their colours, both photos were AI-restored, and 2 of 3 structures were redrawn (benzene, and the stereocentre of the hydroxy acid as drawn); the heavily speckled amide was cleaned but not redrawn, because the reader was not confident enough.

## Picture groups (classification)

After the pictures are extracted, each one is classified, so you can review chemical diagrams, flowcharts and graphs, tables and other images separately.

- **Extraction first, unchanged.** Pictures are taken out by the existing extraction code, without any loss: Word image parts are copied byte for byte, PDF images are written as stored (JPEG streams unchanged, other images pixel for pixel), image files are read as they are. The classifier only classifies these files; it never extracts or changes anything. Tests check both.
- **The classifier.** [DocumentFigureClassifier v2.5](https://huggingface.co/docling-project/DocumentFigureClassifier-v2.5) (docling-project, MIT licence; an EfficientNet-B0 trained on figures from PDF documents) recognises 26 figure types. It runs with ONNX Runtime on the processor in a few milliseconds per picture, from a 16.9 MB model installed by `worker/setup_ai.py` (checked against its SHA-256; nothing is downloaded while the app runs).
- **Groups.** The figure types are grouped: *chemistry structure* → Chemical diagrams; *flow chart, line chart, bar chart, pie chart, scatter plot, box plot, engineering drawing* → Flowcharts and graphs; *table* → Tables; everything else (photograph, logo, icon, screenshot, map, signature, stamp, ...) → Other images. The confidence shown is the total probability of the group's figure types; below 60 % the suggestion is marked *unsure*. A picture that looks chemical at least 35 % is kept with the chemical diagrams, so a chemist sees it rather than it being lost among the other images.
- **Sheets of several figures** (framed panels) have each panel classified on its own, when the classifier is not already sure the whole picture is one figure (75 %): the cells of a table and the boxes of a flowchart are not separate figures. A sheet is listed under every group its panels belong to.
- **Measured on real pictures:** a 91-picture quality document: tables 0.99–1.00, chemical structures 0.74–1.00, flowcharts and charts 0.78–1.00, the company logo 1.00; real photos 0.90–0.96; sheets of mixed figures 0.44–0.68 as a whole, with each panel classified correctly (structures, photos, tables, charts).
- **MolScribe as a second check, chemical diagrams only.** For a picture (or panel) classified as a chemical diagram, MolScribe tries to read its atoms and bonds. It is never run on tables, graphs, photos or other images, and the chemistry redraw is only ever tried on chemical diagrams. When MolScribe reads it confidently (at least 70 %), in one piece, with its bonds found as lines in the picture, and RDKit accepts it as a molecule, the picture shows *MolScribe read 7 atoms and 6 bonds (C5H11NO2S)*. This is a reading, not a verified structure. When it cannot read it reliably, the picture **stays with the chemical diagrams** and is marked **Possible chemical diagram — needs review**, with the reason.
- **Classified as chemical is not chemical structure verified.** The app keeps three things apart:
  - *Classified as a chemical diagram*: the classifier's suggestion (or your choice);
  - *MolScribe read N atoms and M bonds*: a second reading, not a verification;
  - *Chemical structure verified*: shown only for a redraw that passed every automatic check (see [Redraw chemical structures](#redraw-chemical-structures)). Even then, the checks compare the redraw with what was read from the picture; they do not prove that it matches the original, so a chemist compares and approves every redraw.
- **Chemical details only for chemical pictures.** Structure checks, chemical errors and redraw details are shown for chemical diagrams only (and for any picture that was actually redrawn). Tables, graphs, photos and other images never show them.
- **Changing a category** to *Chemical diagram* runs MolScribe's check on that picture. When redraw is switched on and a change decides whether a picture may be redrawn, the picture is enhanced again; its decision is cleared, as for any new result. No chemical redraw is ever approved automatically, and neither is a possible chemical diagram: *Approve safe ones* leaves both for you.
- **When the classifier is not installed**, pictures are listed as *Not classified* with the other images, the review screen says so, and the chemistry redraw is not used.

## Optional AI features

These are installed once with `python worker/setup_ai.py` (about 3 GB of free disk space is needed during setup). After setup, everything works offline.

### Try AI after processing

Every picture is first processed with the normal OpenCV steps. AI is an optional extra you try afterwards on the review screen; it never runs by itself.

- **Suggestions first.** Before you try anything, the **AI enhancement (optional)** box under the picture says what was measured and which model may help. It never promises an improvement.

  | Measured | Suggestion | Model |
  |---|---|---|
  | grain from a scan or camera | *Grain detected — SCUNet may help.* | SCUNet |
  | JPEG saved at quality 60 or lower | *Strong JPEG compression detected — FBCNN may help.* | FBCNN |
  | blurred text (1.6 px or more) | *Blurred text detected — DocRes may help (every letter it shows is checked against the original).* | DocRes |
  | a soft or noisy photo in *Other images* | *Soft or noisy photo — Real-ESRGAN may make it look clearer (it can add fine texture, so compare closely).* | Real-ESRGAN |
  | nothing of the above | *No suitable AI repair detected.* | — |

  The photo model is never used on text, tables, flowcharts, graphs or chemical diagrams, and never when a faithful repair (grain, JPEG, text) applies. Sheets of several figures, redrawn structures and black-and-white pictures get no AI. A model that is not installed is named with *is not installed, so it cannot be tried (run worker/setup_ai.py)*.
- **One picture:** **Try AI enhancement** makes an AI version of the shown picture and, when it passes the checks, applies it.
- **The whole job:** **Try AI on eligible images (N)** at the top tries only the pictures with an installed, suitable model (N counts them), with a progress bar and **Stop**. A picture used in several places is processed once, and every place gets the same AI version. Usable AI versions are applied to the pictures you have not decided yet, and the note at the top says how many were applied, how many did not pass the checks and how many you had already decided. Pictures without a suitable repair are left unchanged.
- **The same checks as every result.** The AI version is made from the original (the AI step, then the remaining normal steps) and must pass the text check (no number and at most 3 % of the characters may read differently, and DocRes may show no digit the original does not show readably), the detail check against the OpenCV result (at most 10 % more detail lost or added), and, for chemical diagrams, the chemical check (MolScribe must read the same molecule before and after). If a check fails, the AI version is shown as *not usable* with the reason, for example *it loses or adds more detail than the OpenCV result*, and the OpenCV result stays in use.
- **Compare three versions.** With an AI version, a **Compare** row above the pictures switches between *Original ↔ OpenCV result*, *Original ↔ AI version* and *OpenCV result ↔ AI version*, side by side or with the slider.
- **Clearly shown when applied.** A usable AI version is applied at once to a picture you have not decided yet. The screen then shows it in purple everywhere: the box under the picture turns purple with **✓ AI enhancement applied (SCUNet)**, the picture's label reads *AI version (SCUNet) · in use*, its badge reads *AI enhanced · needs review*, its row in the list reads *✓ AI enhancement applied (SCUNet)*, and a note at the top of the list says *✓ AI enhancement applied to photo.png (SCUNet)*. When it is not applied, the box says why: *AI enhancement not applied (SCUNet)* with the failed check, *AI version ready · not in use*, or *AI enhancement could not run* (for example a missing model). While AI runs, the box shows a progress bar.
- **You decide, and can switch back.** **Version to keep** switches between the OpenCV result and a usable AI version at any time; the OpenCV result is always kept. Trying AI never approves a picture. For a picture you already approved (or chose to keep as the original), AI does not change anything by itself: after **Try AI enhancement** the app asks straight away *Use the AI version instead? It will then need your approval again* (*Yes, switch* / *No, keep my approval*). **Approve safe ones** never approves an AI version: approve it picture by picture.
- **Saved with the job:** the AI version and your choice are kept when you come back later. The new copy uses the version you chose, and the report says which one it was. Trying another strength makes a new OpenCV result, so its AI version is discarded and can be tried again.

### Graphics chips (GPU)

- **What it does:** on Windows the setup installs the DirectML build of ONNX Runtime (MIT). The AI models then run on any NVIDIA, AMD or Intel graphics chip, and on the processor when there is none. Clients install nothing else.
- **Which chip:** Windows often lists the slow built-in chip first, so the chip with the most memory of its own is used.
- **Each model is tried once per computer.** The first time, a model runs a short trial on the graphics chip in a separate process, so a driver crash cannot stop the app, and is timed against the processor. The faster device is remembered in `models/device_choice.json`. If the chip later fails, the model falls back to the processor.
- **Measured on a laptop (Ryzen 5 7535HS, RTX 3050, built-in Radeon):**

  | Photo model, one 1000×700 picture | Time |
  |---|---|
  | Processor | 11.0 s |
  | Built-in Radeon chip | 2.1 s |
  | RTX 3050 | 0.9 s |

  The results were the same on every device (largest difference 0.004 of 255 levels). On the built-in Radeon chip, SCUNet crashed the graphics driver at variable sizes and was slower than the processor at a fixed size, so it stays on the processor there; the other models use the chip.
- **Laptops on battery** often switch the graphics card off; the app then uses the built-in chip or the processor.
- Set `IQE_AI_DEVICE=cpu` to always use the processor.

### AI repair

- **How to use it:** **Try AI enhancement** on the review screen, after the normal processing (see [Try AI after processing](#try-ai-after-processing)).
- **What it does:** where needed, the image is also repaired by AI, and the measured best version is kept:

  | Model | Repairs | Tried when | Licence |
  |---|---|---|---|
  | SCUNet | grain from scans and cameras | grain level above 4 | Apache-2.0 |
  | FBCNN | JPEG blocks and ringing | JPEG saved at quality 60 or lower (read from the file), also when no other step is needed (then compared with the unchanged original) | Apache-2.0 |
  | DocRes | blurred document text | text blurred by 1.6 px or more, and readable in the original | MIT |

- **How the best version is chosen:**
  - **Text:** each version is read by OCR on the original's lines. A version is ruled out if a number or more than 3 % of the characters read differently from the original. The most legible remaining version wins; between equally legible ones, the one with less grain, then the simpler one.
  - **AI-sharpened text** (DocRes) is kept only if it shows no digit, and almost no letter, that the original does not show readably. The sharpened version's own lines are read too, so text that appears where the original shows nothing readable is caught. Reason: on a page blurred beyond recognition, DocRes produced sharp but wrong digits (22.4 % → 22.6 %, 1,240 → 1,265).
  - **Pictures without readable text:** an AI version may lose or add at most 10 % more detail than the filters' version, and wins when it has clearly less grain.
- **Shown in Details and the report:** which version was kept, why the others were not, and where each model ran.
- **Measured (characters read correctly by OCR, synthetic labels with known text):**

  | Damage | Filters only | With AI repair |
  |---|---|---|
  | Heavy grain | 95.1 % | 100 % |
  | Severe JPEG compression | 98.8 % | 100 % |
  | Blur 2.5 px | 92.7 % | 97.6 % (DocRes) |
  | Blur 3.0 px | 26.8 % | 91.5 % (DocRes) |

  On the blurred labels DocRes was still ruled out when the blurred original was read consistently wrong (a "B" read as "8"): the tool cannot tell a corrected reading from an invented one, so it never risks a digit.
- **Not used:** NAFNet's published de-blur models are trained on camera shake. On out-of-focus text they read worse than the filters (12 of 25 words against 25 of 25), so they were left out.
- **Speed:** SCUNet takes about 30 s for a 1000×700 picture on a laptop processor; DocRes about 2 s per 384-pixel tile on a built-in graphics chip. AI repair only runs when you try it, and only where its trigger is measured.
- **Model files:** SCUNet and FBCNN come from their authors' GitHub releases. DocRes's authors publish on OneDrive, which now needs a sign-in, so the setup uses an identical re-hosted copy on Hugging Face, pinned by SHA-256; an official copy can be given with `--docres-file`. All are converted once to ONNX and checked against the original model (largest difference 0.01 of 255 levels).

### Text check (OCR)

- **What it does:** images with text are read before and after enhancement by RapidOCR (Apache-2.0), and the two readings are compared.
- **Reader:** PP-OCRv5 detection and English recognition (Apache-2.0, about 13 MB, checked against their SHA-256), with the package's PP-OCRv3 as the fallback. On small scanned document text (6–9 pt at 100–150 dpi, three fonts) PP-OCRv5 read 86.9 % of the characters correctly against 83.3 % for PP-OCRv3; on 7 pt scans 97–99 % against 82–95 %. The same reader checks atom labels for chemical redraws.
- **What it compares:**
  - The lines are found once in the original and read at the same places in both images.
  - Only characters OCR reads the same way twice in the original count (lines that differ get extra readings).
  - Punctuation is ignored.
- **When an image is flagged:** if any digit changes, or more than 3 % of the characters read differently, the image is flagged and rated *Check*.
- **Ink check:** a claimed changed digit is verified against the pixels around that character. If its strokes are all still there and nothing new was drawn, OCR merely read a symbol differently (a ₹ sign read as "7" in the blurry original) and the image is not flagged; the reading is reported as uncertain. A truly replaced digit (4521 → 4527) loses about 10 % of the edges in its window and is flagged.
- **Measured:**
  - An unchanged label passes.
  - A changed batch number (4521 → 4527) is caught.
  - Time is about 1–4 s per image with text, and about 21 s for a dense full page (at most 60 lines are read).

### Redraw chemical structures

- **Only for chemical diagrams.** Redraw (and so MolScribe) is only tried on pictures, or panels, classified as chemical diagrams (see [Picture groups](#picture-groups-classification)); tables, graphs, photos and other images are never redrawn.
- **How to use it:** choose *Chemical structures: Redraw when verified* on the start page. Redraw jobs use one worker, because the reader needs about 0.8 GB of memory.
- **How a structure is redrawn:**
  1. MolScribe (MIT) reads the drawing.
  2. RDKit (BSD) builds the molecule. Condensed labels (COOH, NH2, …) are expanded from a fixed table; an unknown label means no redraw. Stereochemistry comes from the drawn wedges and the geometry.
  3. It is then redrawn in the original layout, at the original pixel size, with the wedges on the same bonds and the labels condensed as in the original.
- **Checks: the image is redrawn only if every one passes. Otherwise it is enhanced as usual, and the reason is shown.**
  - The reader's confidence.
  - The reader's own SMILES has the same connectivity.
  - The reading, aligned onto the drawing, explains it: every bond is drawn as read (single, double, triple, solid or hashed wedge), and at least 93 % of the ink is explained.
  - Hashed wedges counted in the picture equal those read.
  - OCR labels exist in the molecule, and there is no other text (captions, logos).
  - A second reading at 0.8× scale gives the identical molecule.
  - **The redraw reads back as the identical molecule** (same InChIKey, including stereochemistry).
- **"Verified", not "proven".** A redraw that passes every check is marked *Chemical structure verified by the automatic checks*. These checks compare the redraw with what was read from the picture; they do not prove that it matches the original. The app never claims that MolScribe or RDKit proved a match.
- **Approval:** every redraw needs a chemist's approval. *Approve safe ones* never includes redraws.
- **Structure checks, available when needed.** For identified chemical candidates, expand **Structure checks** to see the failed checks in plain words. **Highlight these areas on the original** turns on numbered boxes where locations are available; highlights and the section reset when you select another image. The gallery keeps a short status instead of a second red mismatch count. Actual redraws still show the required chemist-review message, including when verification is incomplete, and the report retains detailed findings. Old confidence-only guesses without chemical classification or a molecular identity are not presented as chemical errors. Examples:
  - *Bond C–C: the AI read a bond here, but no line is drawn at this place.*
  - *Bond C–N: the line gets wider like a bold wedge (3D bond), but the AI read a plain bond.*
  - *The drawing has 1 dashed wedge (3D bonds), but the AI read 0. A redraw would change the 3D form (for example D/L).*
  - *Text "Figure 3 Aspirin tablets" in the picture is not part of the structure.*
  - *The AI is not sure about this atom (N, 64 % sure).*
  - *The clean redraw does not read back as the same molecule.*
- **Drawings with written groups (HOOC, COOH, NH2, OH) and scanned lines.** The drawing check now:
  - treats the atoms the reader places beside a written group (often just outside its text) as part of that label;
  - looks for a bond into a label between the atom and the label's edge, close to the read direction, because the line ends at the letter it joins, not where the reader put the atom;
  - fits a bond whose read line misses the ink to the drawn line within 15 % of a bond length (thick scanned lines sit a little beside the reading);
  - measures a bond's width only on its own stroke, a quarter and three quarters along it, so a crowded junction no longer looks like a wedge;
  - counts a thick shape as a solid wedge only if it runs along a bond for at least a fifth of its length, and never inside a label;
  - when a redraw does not fit the picture, tries thinner lines, smaller labels and a fresh layout first.
  
  Measured on a 23-image drug-substance Word file (18 structures): 1 redrawn before, **13 redrawn and proven** now. The 5 still refused are correct: in 2 the reader missed a drawn hashed wedge (the redraw would lose the configuration), in 3 it placed a bond where none is drawn.
- **A blurred original drawn unclearly:** blur makes lines and letters look thicker, and blurred atom positions can fold a bond onto its neighbour. When the redraw does not read back as the same molecule, it is drawn again with thinner lines and smaller labels, then laid out afresh by RDKit and turned to the original's orientation (never mirrored; wedges are placed again from the molecule's stereochemistry). It is kept only if it then reads back as the identical molecule. Measured: a very blurred vanillin drawing, refused before, is now redrawn and proven.

### Redraw all

- **How to use it:** choose *Chemical structures: Redraw all* on the start page.
- **Ordinary images stay ordinary.** Text and table cells are checked for drawing strokes outside their labels before chemical recognition. Images that are not chemical candidates receive normal enhancement without chemical mismatch messages, red boxes or chemist-approval warnings. Actual chemical redraws still show all required checks.
- **What it does:** every structure the reader can read is redrawn. A redraw that fails a check about certainty (reader confidence, bonds or wedges in the picture, OCR labels, the second reading, the read-back, agreement with the original) is still made, and marked **not verified** with the checks it failed.
- **Still never redrawn:** pictures that are not chemical structures (reader confidence below 0.30: tables, text, photos, logos), readings that give no molecule or explain less than 60 % of the drawing, and pictures with other text such as captions, which a redraw would remove.
- **Never taken for a structure** (no redraw, no atom list, no chemist warning), in every mode:
  - pictures where the bonds the reader saw are not drawn lines: fewer than 2 bonds, or fewer than half of them, are found in the picture. Regression: a table cell holding "0.9999" was read as two O atoms and, with *Redraw all*, replaced by a drawn ring;
  - readings in more than two separate pieces, which is the reader guessing at a text box;
  - an unsure reading (below 0.70) of a picture with ordinary words, such as a flowchart box "Manufacturing Process Steps".
- **Redraws that are not verified** are shown as *Redrawn · not verified*, rated *Check* instead of *Verified*, flagged for review, and never included in *Approve safe ones*.
- **Measured on five blurred, grainy or compressed test drawings:** aspirin, paracetamol, ibuprofen and vanillin were redrawn and proven; caffeine (a noisy scan with specks) was redrawn correctly but not proven.
- **Noisy drawings** are read again after cleaning; the original read directly must agree on the molecule.
- **Measured on a real 19-image quality dossier:**
  - D-penicillamine (S) and L-penicillamine (R) drawings were redrawn with the correct configuration.
  - MolScribe's own SMILES had flipped the stereocentre of the D-form (bold-wedge) drawings, and the redraw corrected that.
  - Drawings where the reader missed a hashed wedge were refused, as were a logo and drawings the reading could not fully explain.

### AI clarity for photos

- **What it does:** Real-ESRGAN's general model (BSD-3-Clause), run with ONNX Runtime (MIT), restores the photo at 4× and reduces it back to its own pixel size.
- **Where it is used:** only when you try AI on a picture in *Other images* that is a soft or noisy photo; never on text, tables, flowcharts, graphs, chemical diagrams or mixed pages. It can add fine texture, so compare it closely with the OpenCV result before choosing it.
- **Measured:** on a noisy, blurred, compressed photo, similarity to the clean original went from 0.51 to 0.63, against 0.56 with the classic steps, in about 3 s for 640×480.

## Enhancement levels

Auto mode has four levels. Each one does more than the one before. Choose the level for the whole job on the start page. On the review screen, click **1–4** to re-process one image at another level.

| Level | What it does |
|---|---|
| **1 · Light** | Minimal changes. Images that are already good stay untouched. |
| **2 · Standard** (default) | Clearly visible improvements (see below). |
| **3 · Strong** | Stronger cleaning and full deblurring. Text on low-quality images becomes slightly bolder. |
| **4 · Maximum** | Everything in Strong, plus fine-detail processing for more images (shown below 300 dpi, or files under 2,000 px), the crispest text and bolder lines. |

- **Already-good images are untouched at every level.** A higher level adds nothing to them.
- **Accuracy is shown for each level.**
  - Each result shows its rating, for example *Accuracy: High · 99.6 % of lines and letters kept*.
  - When you try several levels, the review screen lists them all.
  - It warns when a lower level was more accurate.
  - A stronger level is not automatically better, so compare zoomed in before approving.

**Bolder text** works on low-quality text, tables, diagrams and chemical structures. It makes thin, faint strokes slightly thicker and darker:

- **How much:** at most one pixel, in the stroke's own colour.
- **Unchanged:**
  - strokes that are already thick
  - photos
  - paper
- **Never closed:** gaps narrower than 3 pixels, such as the holes in "e", "a" and "8", double bonds and close table lines. Nothing new is drawn, and no pixel gets lighter.
- **Setting:** *Automatic* (levels 3 and 4, low-quality images only), *On* or *Off*. Change it on the start page (**More options**) or under **Fine-tune (advanced)**.

What **Standard** does, by type of image:

- **Scanned pages and printed text.**
  - Uneven lighting and yellowed or grey paper are evened out to a clean white background, and faded ink becomes dark.
  - Photocopy speckle and grey shading are smoothed away.
  - Only tiny, faint, dot-like specks are ever removed. Letters, digits, subscripts, punctuation and lines are never deleted, even when they are grey.
  - Pastel backgrounds (coloured labels, table headers) keep their colour.
- **Blurred images are deblurred.** The blur is measured, then reversed by deconvolution; text is also given crisp edges. On a blurred table with a known sharp original, similarity to that original rose from 0.81 (enlarging only) to 0.91–0.95.
  - **Blur up to about 1.6 px** is recovered reliably: the text comes back crisp and correct.
  - **Heavier blur** has lost the fine parts of letters, and full recovery can produce wrong characters (for example "B" read as "D"). Standard therefore removes only part of that blur, leaves edges soft and flags the image "heavy blur". **Strong** and **Maximum** recover it fully, which is crisper, and still flag it so you can check the characters.
- **Mixed pages** (photos next to text, tables, charts or structures) are processed region by region:
  - Document areas are cleaned, photo areas get photo treatment, and flat labels and frames stay unchanged.
  - Photos of bright textured objects (crystals, packaging, shelves) are recognised and never "cleaned" like paper.
  - Very faint printed lines can become fainter. Results like this are flagged for review.
- **Low-resolution images: fine-detail processing.** The image is enlarged *inside the program* (2× or 1.5×), and deblurring, sharpening, crisp text and bolder text run on that finer grid. The result is then reduced back to the original pixel size. Edges come out smoother and cleaner, and **the output always has exactly the original pixel size**. This applies to:
  - *Documents:* images shown below 150 dpi.
  - *Uploaded image files:* images smaller than 1,000 px (2×) or 1,600 px (1.5×).
- **Photos.** Only measured problems are repaired: real noise (fine texture is recognised and kept), blur, strong JPEG compression (AI repair) and fading. A faded photo gets each colour's own black and white point back, which also removes the yellow or magenta cast of the fading. On damaged test photos measured against the true picture, the colour error of faded photos fell from 17–27 to 6–9 (ΔE), and no photo came out further from the true picture than its damaged input. Local contrast changes how a photo looks, so it is only added at **Strong** and **Maximum**.
- **Diagrams, chemical structures and charts.** Shapes and colours are never altered. Clean digital ones are kept as they are; blurred ones are deblurred. Charts on scanned or faded paper are cleaned with their bars, colours and grid lines kept.

## Phase 8 results

### Real documents

Test set: 4 Wikipedia article PDFs (Aspirin, Caffeine, Penicillin, Microscope), a public Word test file, and a simulated poor scan.

- All exports passed validation.
- No image failed.
- Peak worker memory stayed under 95 MB.

Problems found on real images, and fixed:

| Problem | Fix |
|---|---|
| Colour chemical structures on black changed colour | No contrast change for dark or colourful diagrams |
| Crystal texture was smoothed as if it were noise | Two-scale noise/texture test |
| An unevenly lit scan was treated as a photo | Lighting is evened out before classifying |

### Scale (generated PDFs)

Every image in these documents needed denoising and 2× enlargement, which is the slowest case. Times are on this development PC, with 1 worker (unless stated), and the test suite was sometimes running at the same time.

| Images | Total time | Average per image | Peak worker memory | Peak app memory | Export |
|---|---|---|---|---|---|
| 10 | 10 s | 0.79 s | 84 MB | 50 MB | valid |
| 100 | 70 s | 0.66 s | 84 MB | 64 MB | valid |
| 500 | 5.7 min | 0.66 s | 85 MB | 86 MB | valid |
| 1,000 | 12 min | 0.70 s | 86 MB | 141 MB | valid |
| 2,000 | 15.8 min | 0.45 s | 86 MB | 232 MB | valid |

**Parallel workers.** With 2 workers, the 500-image document took 2.8 min instead of 5.7 min (about 2× faster), for about 16 MB more memory. 2 workers is therefore the default on machines with at least 4 CPU cores; smaller machines use 1.

Memory stays flat as documents grow, because images are processed one at a time, results are kept on disk, PDFs are handled in chunks, and the gallery is virtualised. Most of the time goes on noise reduction. Clean images take about 0.05–0.2 s each.

PDF and DOCX file handling on its own (scan, extract, replace and validate, without enhancement) was measured earlier:

| Document | Scan | Extract | Replace + validate | Peak memory |
|---|---|---|---|---|
| 500-page, 1,000-image PDF (257 MB) | 1.1 s | 3.9 s | 6.9 s | 197 MB |
| 1,000-image DOCX (385 MB) | 0.5 s | 2.7 s | 10.4 s (export + validation) | 65 MB |
