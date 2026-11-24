# Dashcam Sign Classifier

An Intelligent System to Classify Road Traffic Signs from Dashcam Images
(Deep Learning & Computer Vision)

Student: Budumuri Mahesh (23ME1A5470)
Guide: Mrs. P. Neeraja
Dataset: GTSRB

## Run it
Open `index.html` in any modern browser. No install or build step.
(Google Fonts load from the internet; the page still works offline with fallback fonts.)

## Files
- `index.html`  page structure and project sheet
- `style.css`   light/dark theme and layout
- `app.js`      dashcam simulator, recogniser, ADAS alert logic, session log

## What the prototype is
The drive is simulated and the sign position comes from the simulator.
The recogniser in `app.js` (function `classify`) is a template-matching stand-in:
it takes a 32x32 crop and returns probabilities for 14 of the 43 GTSRB classes.
The trained CNN is planned, not included. The 95% accuracy figure is a target.

## Planned CNN (Keras)
Input 32x32x3 -> Conv32 -> Conv32 -> MaxPool -> Dropout .25
-> Conv64 -> Conv64 -> MaxPool -> Dropout .25 -> Flatten
-> Dense256 -> Dropout .5 -> Dense43 softmax (1,125,451 parameters)

## Swapping in the real model
Train on GTSRB, export with TensorFlow.js, and replace the body of `classify()`
so it returns the 43 class probabilities from the model for the 32x32 crop.
