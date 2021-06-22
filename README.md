# Arknights Team Recognizer
## 明日方舟队伍识别器

This project is still working in progress. Contact me if you are willing to help on OCR and image comparison.

## Idea

This application is a part of ArkRecordWiki and developed by React. Since it is much more complicated in logic and independent in functionality, I make it a separate repo.

The total goal of this application is to recognize the team members and their skill from a screenshot. It could alleviate the burden of writing down all the name and skills of operators, especially for teams like 3-star and E1-Level1 teams which usually carry more operators than other categories.

## Process

- Line Detection
  - Canny
  - Hough Transform
  
- Image Segmentation
  - Unsupervised clustering to sort the lines
  - Calculate dimension of gaps and char cards
  - Render to canvas
  
- OCR
  - Tesseract.js (poor result)
  - PaddlePaddle OCR (Great but run in python env, in progress)
  - Get the operator names 
  
- Skill recognition
  - Fetch skill icons based on operator name
  - Image comparison (in progress)

## Libraries

- GammaCV
  - A fantastic library provides Hough Transform function and other CV methods, which use tensor and gpu to accelerate the process
  
- Simple-statistics
  - A light-weighted library to do some math. It offers great functions to generate 1-D clustering results (i.e. natural breaks)
  
- Tesseract.js
  - It is quite convenient that it provides API for users, and the package itself is just some simple wrapper, but the result for Chinese is not satisfying, due to which it will be replaced later.
  
- Paddle OCR
  - An open-source OCR library for mainly Chinese, composed of DB text detection, detection frame correction and CRNN text recognition with high accuracy but low space occupation. Everything perfect but not designed for a node server. I would like to take some time to adopt it. 
  
- Resemble.js
  - The famous image comparison library to calculate similarity of two images. Working in progress.
  
## License

This project is using MIT license. 

This sample images of the application is provided by reaving, Matsuka__, Terpenes, mirrorMK, Function____ and 卓荦zoro which will **NOT** be sharing.