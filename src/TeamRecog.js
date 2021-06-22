import * as gm from "gammacv";
import React, { useEffect, useState } from "react";
import { mean, ckmeans, variance } from "simple-statistics";
import { createWorker } from "tesseract.js";
import { getImageData } from "gammacv";
import "bootstrap/dist/css/bootstrap.min.css";

export default function TeamRecog() {
  const [ori, setOri] = useState();
  const [hough, setHough] = useState(false);
  const [low, setLow] = useState(0.6);
  const [high, setHigh] = useState(1);

  const [gapH, setGapH] = useState();
  const [gapHOffset, setGapHOffset] = useState(1.5);
  const [gapW, setGapW] = useState();
  const [charW, setCharW] = useState();

  const [startOCR, setStartOCR] = useState(false);
  const [ocr, setOcr] = useState();
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState(0);

  async function OCR() {
    if (ocr) return;
    setStartOCR(true);
    const canvasR = document.querySelectorAll("canvas")[1];
    const worker = createWorker({
      logger: (m) => {
        setProgress(parseFloat(m.progress));
        setStatus(m.status);
        console.log(m);
      },
    });
    await worker.load();
    await worker.loadLanguage("chi_sim");
    await worker.initialize("chi_sim");
    const result = await worker.recognize(canvasR);
    console.log(result);
    setOcr(result.data.text);
  }

  function reset() {
    console.clear();
    const canvas = document.querySelectorAll("canvas")[2];
    const ctx = canvas.getContext("2d");
    ctx.drawImage(ori, 0, 0);
  }

  async function setSample(index) {
    if (index === "None") return;
    const image = new Image();
    image.onload = () => {
      const canvas = document.querySelectorAll("canvas")[2];
      canvas.width = image.width;
      canvas.height = image.height;
      canvas.getContext("2d").drawImage(image, 0, 0);
      setOcr(undefined);
      setOri(image);
    };
    image.src = "/images/team" + index + ".jpg";
  }

  function handleUpload(evt) {
    const reader = new FileReader();
    if (!evt.target.files[0]) return;
    reader.readAsDataURL(evt.target.files[0]);
    reader.onload = (evt) => {
      const image = new Image();
      image.onload = () => {
        const canvas = document.querySelectorAll("canvas")[2];
        canvas.width = image.width;
        canvas.height = image.height;
        canvas.getContext("2d").drawImage(image, 0, 0);
        setOcr(undefined);
        setOri(image);
      };
      image.src = evt.target.result.toString();
    };
  }

  useEffect(() => {
    const canvasAll = document.querySelectorAll("canvas");
    canvasAll[0].height = 0;
    canvasAll[1].height = 0;
  }, []);

  useEffect(() => {
    if (hough) {
      reset();
      const canvas = document.querySelectorAll("canvas")[2];
      const input = new gm.Tensor("uint8", [canvas.height, canvas.width, 4]);
      // image data to tensor
      gm.canvasToTensor(canvas, input);

      // build pipeline
      let operation = gm.grayscale(input);
      operation = gm.downsample(operation, 2, "nearest");
      operation = gm.gaussianBlur(operation, 3, 3);
      operation = gm.sobelOperator(operation);
      operation = gm.cannyEdges(operation, low, high);
      operation = gm.pcLines(operation, 2, 2, 2);
      const output = gm.tensorFrom(operation);

      // run the session
      const sess = new gm.Session();
      sess.init(operation);
      sess.runOp(operation, 0, output);

      const maxP = Math.max(canvas.height, canvas.width);
      let lines = [];

      for (let i = 0; i < output.size / 4; i += 1) {
        const y = Math.floor(i / output.shape[1]);
        const x = i - y * output.shape[1];
        const value = output.get(y, x, 0);
        const x0 = output.get(y, x, 1);
        const y0 = output.get(y, x, 2);

        if (value > 0.0) {
          lines.push([value, x0, y0]);
        }
      }

      // sort by line value
      const numOfLines = 400;
      lines = lines.sort((b, a) => a[0] - b[0]).slice(0, numOfLines);

      const horizontal = [];
      const vertical = [];

      for (let i = 0; i < lines.length; i += 1) {
        let line = new gm.Line();
        line.fromParallelCoords(
          lines[i][1] * 2,
          lines[i][2] * 2,
          canvas.width,
          canvas.height,
          maxP,
          maxP / 2
        );
        if (Math.abs(line.angle - 90) > 89) {
          // ~0
          horizontal.push(line);
        } else if (Math.abs(line.angle - 90) < 1) {
          // ~90
          vertical.push(line);
        }
      }

      // sort horizontal lines by distance with center
      horizontal.sort(
        (lineA, lineB) =>
          Math.abs((lineA.y1 + lineA.y2) / 2 - canvas.height / 2) -
          Math.abs((lineB.y1 + lineB.y2) / 2 - canvas.height / 2)
      );

      // get the first line nearest center
      const hL = horizontal[0];
      // gm.canvasDrawLine(canvas, hL, "rgba(0, 255, 0, 1.0)");

      // gapH is the gap distance between upper char and lower char cards
      const tempGapH = Math.abs(hL.y1 + hL.y2 - canvas.height) + gapHOffset;
      setGapH(tempGapH);
      console.log("gapH height", tempGapH);

      // draw its mirror
      // let mirrorY;
      // if (hL.y1 > canvas.height / 2) {
      //   mirrorY = (hL.y1 + hL.y2) / 2 - gapH;
      // } else {
      //   mirrorY = (hL.y1 + hL.y2) / 2 + gapH;
      // }
      // const mirror = new gm.Line([hL.x1, mirrorY, hL.x2, mirrorY]);
      // gm.canvasDrawLine(canvas, mirror, "rgba(0, 255, 0, 1.0)");

      // for vertical lines, it is represented by a intersection formula
      // and the middle point has to be calculated by a intersection function
      // refer to https://github.com/PeculiarVentures/GammaCV/issues/120
      const crossLine = new gm.Line(
        0,
        canvas.height / 2,
        canvas.width,
        canvas.height / 2
      );
      const xCorArray = vertical.map(
        (line) => gm.Line.Intersection(line, crossLine)[0]
      );
      // for (let x of xCorArray) {
      //   const line = new gm.Line(x, 0, x, canvas.height);
      //   gm.canvasDrawLine(canvas, line, "rgba(0, 0, 255, 1.0)");
      // }

      // use ckmeans to find the natural breaks
      let xCorBreaks;
      for (let i = 2; i < xCorArray.length; i++) {
        const breaks = ckmeans(xCorArray, i);
        let totalVariance = 0;
        for (let cluster of breaks) {
          totalVariance += variance(cluster);
        }
        // unsupervised learning
        if (totalVariance < 20) {
          xCorBreaks = breaks;
          break;
        }
      }
      // console.log("x cor breaks", xCorBreaks);

      // calc and sort all x cors
      const xCors = xCorBreaks
        .map((cluster) => mean(cluster))
        .sort((a, b) => a - b);

      // calc difference between one and another
      const diffs = [];
      for (let i = 0; i < xCors.length - 1; i++) {
        for (let j = i + 1; j < xCors.length; j++) {
          diffs.push(xCors[j] - xCors[i]);
        }
      }

      // use ckmeans again to find the natural breaks
      let diffBreaks;
      for (let i = 2; i < diffs.length; i++) {
        const breaks = ckmeans(diffs, i);
        let totalVariance = 0;
        for (let cluster of breaks) {
          totalVariance += variance(cluster);
        }
        // unsupervised learning
        if (totalVariance < 20) {
          diffBreaks = breaks;
          break;
        }
      }
      const candidateWidth = diffBreaks
        .sort((a, b) => b.length - a.length)
        .map((item) => mean(item));
      console.log("candidate gapW", candidateWidth);

      // empirical evidence
      const emp = {
        gapW: 2 * tempGapH,
        charW: 8 * tempGapH,
      };
      console.log(emp);
      let tempGapW, tempCharW;
      for (let num of candidateWidth) {
        if (!tempGapW && variance([num, emp["gapW"]]) < emp["gapW"]) {
          tempGapW = num;
        } else if (!tempCharW && variance([num, emp["charW"]]) < emp["charW"]) {
          tempCharW = num;
        } else if (tempGapW && tempCharW) {
          break;
        }
      }
      setGapW(tempGapW);
      setCharW(tempCharW);
      console.log("gapW width", tempGapW, "char width", tempCharW);
    }
  }, [hough, ori, low, high, gapHOffset]);

  useEffect(() => {
    // extract text
    if (gapW && gapH && charW) {
      const canvas = document.querySelectorAll("canvas")[2];
      const canvasR = document.querySelectorAll("canvas")[1];
      const ctxR = canvasR.getContext("2d");

      const xStart = canvas.width / 2 - 3.5 * charW - 3 * gapW;
      const textHeight = 2.7 * gapH;
      const yLine1 = canvas.height / 2 - textHeight;

      canvasR.height = textHeight * 2;
      canvasR.width = charW * 6;

      let x = xStart,
        y = yLine1;
      for (let i = 0; i < 12; i++) {
        const textData = getImageData(canvas, x, y, charW, textHeight);
        x += charW + gapW;
        if (i === 5) {
          x = xStart;
          // another empirical parameter that the ratio char card is fixed
          y += charW * 2.336;
        }
        if (i < 6) {
          ctxR.putImageData(textData, i * charW, 0);
        } else {
          ctxR.putImageData(textData, (i - 6) * charW, textHeight);
        }
      }

      // invert color
      const whiteTensor = new gm.Tensor("uint8", [
        canvasR.height,
        canvasR.width,
        4,
      ]);
      whiteTensor.data.fill(255);
      const imageTensor = new gm.Tensor("uint8", [
        canvasR.height,
        canvasR.width,
        4,
      ]);
      // image data to tensor
      gm.canvasToTensor(canvasR, imageTensor);
      const operation = gm.sub(whiteTensor, imageTensor);
      const output = gm.tensorFrom(operation);

      const sess = new gm.Session();
      sess.init(operation);
      sess.runOp(operation, 0, output);

      gm.canvasFromTensor(canvasR, output);
    }
  }, [gapW, gapH, charW]);

  useEffect(() => {
    // extract skill icons
    if (gapW && gapH && charW) {
      const canvas = document.querySelectorAll("canvas")[2];
      const canvasT = document.querySelectorAll("canvas")[0];
      const ctxT = canvasT.getContext("2d");

      const iconSize = charW / 3.4516;
      const xStart = canvas.width / 2 - 2.865 * charW - 3 * gapW;
      const yLine1 = canvas.height / 2 - charW / 1.671;

      canvasT.height = iconSize;
      canvasT.width = iconSize * 12;

      let x = xStart,
        y = yLine1;
      for (let i = 0; i < 12; i++) {
        const textData = getImageData(canvas, x, y, iconSize, iconSize);
        x += charW + gapW;
        if (i === 5) {
          x = xStart;
          // another empirical parameter that the ratio char card is fixed
          y += charW * 2.336;
        }
        ctxT.putImageData(textData, i * iconSize, 0);
      }
    }
  }, [gapW, gapH, charW]);

  return (
    <div className="container py-4">
      <h2 className="mb-4">Arknights Team Recognizer</h2>

      <div className="mb-4">
        <h5 className="mb-2">Step 1: choose an image</h5>
        <label className="form-label">Upload your own team screenshot</label>
        <input
          className="form-control mb-3"
          type="file"
          accept="image/*"
          onChange={handleUpload}
        />
        <div className="mb-3">
          <label className="form-label">
            Or use our provided sample images
          </label>
          <select
            className="form-select"
            onChange={(evt) => setSample(evt.target.value)}
          >
            <option value="None">None</option>
            {new Array(7).fill(0).map((item, index) => (
              <option value={index} key={index}>
                {"Team " + index}
              </option>
            ))}
          </select>
        </div>
        <button
          className="btn btn-outline-primary me-3"
          onClick={() => setHough(true)}
        >
          Split
        </button>
      </div>

      {hough ? (
        <div className="mb-4">
          <h5>
            Step 2: adjust the params to render a correct crop of op names then
            start OCR
          </h5>
          <div className="d-flex">
            <label className="form-label me-3">
              Canny Low Threshold
              <input
                className="form-control"
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={low}
                onChange={(evt) => setLow(parseFloat(evt.target.value))}
              />
            </label>
            <label className="form-label me-3">
              Canny High Threshold
              <input
                className="form-control"
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={high}
                onChange={(evt) => setHigh(parseFloat(evt.target.value))}
              />
            </label>
            <label className="form-label me-3">
              Gap H offset
              <input
                className="form-control"
                type="number"
                min={-5}
                max={5}
                step={0.5}
                value={gapHOffset}
                onChange={(evt) => setGapHOffset(parseFloat(evt.target.value))}
              />
            </label>
          </div>
          <button className="btn btn-outline-primary me-3" onClick={OCR}>
            OCR
          </button>
        </div>
      ) : null}

      {hough && startOCR ? (
        <div className="mb-4">
          <div className="form-label mb-3">{"Status: " + status}</div>
          <label className="form-label me-3s">Progress</label>
          <div className="progress mb-3">
            <div
              className="progress-bar progress-bar-striped progress-bar-animated"
              role="progressbar"
              style={{ width: progress * 100 + "%" }}
            >
              {progress * 100 + "%"}
            </div>
          </div>
          {ocr ? (
            <div>
              Result: <strong>{ocr}</strong>
            </div>
          ) : null}
        </div>
      ) : null}
      <div>
        <canvas />
      </div>
      <div>
        <canvas />
      </div>
      <div>
        <canvas />
      </div>
    </div>
  );
}
