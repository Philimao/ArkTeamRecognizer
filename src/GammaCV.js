import * as gm from "gammacv";
import React, { useEffect, useState } from "react";
import { mean, mode, ckmeans, variance } from "simple-statistics";

export default function GammaCV() {
  const [ori, setOri] = useState();
  const [hough, setHough] = useState(false);
  const [low, setLow] = useState(0.6);
  const [high, setHigh] = useState(1);

  const [gapH, setGapH] = useState();

  function reset() {
    console.clear();
    const canvas = document.querySelectorAll("canvas")[0];
    const canvasR = document.querySelectorAll("canvas")[1];
    const ctx = canvas.getContext("2d");
    const ctxR = canvasR.getContext("2d");

    ctx.drawImage(ori, 0, 0);
    ctxR.translate(ori.height / 2, ori.width / 2);
    ctxR.rotate(Math.PI / 2);
    ctxR.drawImage(ori, -ori.width / 2, -ori.height / 2, ori.width, ori.height);
    ctxR.rotate(-Math.PI / 2);
    ctxR.translate(-ori.height / 2, -ori.width / 2);
  }

  function handleUpload(evt) {
    const canvas = document.querySelectorAll("canvas")[0];
    const canvasR = document.querySelectorAll("canvas")[1];
    const ctx = canvas.getContext("2d");
    const ctxR = canvasR.getContext("2d");

    const reader = new FileReader();
    if (!evt.target.files[0]) return;
    reader.readAsDataURL(evt.target.files[0]);
    reader.onload = (evt) => {
      const image = new Image();
      image.onload = () => {
        canvas.width = image.width;
        canvas.height = image.height;

        canvasR.width = image.height;
        canvasR.height = image.width;

        ctx.drawImage(image, 0, 0);

        ctxR.translate(image.height / 2, image.width / 2);
        ctxR.rotate(Math.PI / 2);
        ctxR.drawImage(
          image,
          -image.width / 2,
          -image.height / 2,
          image.width,
          image.height
        );
        ctxR.rotate(-Math.PI / 2);
        ctxR.translate(-image.height / 2, -image.width / 2);
        setOri(image);
      };
      image.src = evt.target.result.toString();
    };
  }

  useEffect(() => {
    function getLinesFromCanvas(canvas) {
      // const canvas = ;
      const input = new gm.Tensor("uint8", [canvas.height, canvas.width, 4]);
      gm.canvasToTensor(canvas, input);

      let operation = gm.grayscale(input);
      operation = gm.downsample(operation, 2, "nearest");
      operation = gm.gaussianBlur(operation, 3, 3);
      operation = gm.sobelOperator(operation);
      operation = gm.cannyEdges(operation, low, high);
      operation = gm.pcLines(operation, 2, 2, 2);
      const output = gm.tensorFrom(operation);

      const sess = new gm.Session();
      sess.init(operation);
      sess.runOp(operation, 0, output);

      return output;
    }

    if (hough) {
      (function processImage() {
        reset();
        const canvas = document.querySelectorAll("canvas")[0];
        const canvasR = document.querySelectorAll("canvas")[1];
        const output = getLinesFromCanvas(canvas);
        const outputR = getLinesFromCanvas(canvasR);

        const maxP = Math.max(canvas.height, canvas.width);
        let lines = [];
        let linesR = [];

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

        for (let i = 0; i < outputR.size / 4; i += 1) {
          const y = Math.floor(i / outputR.shape[1]);
          const x = i - y * outputR.shape[1];
          const value = outputR.get(y, x, 0);
          const x0 = outputR.get(y, x, 1);
          const y0 = outputR.get(y, x, 2);

          if (value > 0.0) {
            linesR.push([value, x0, y0]);
          }
        }

        const numOfLines = 400;

        lines = lines.sort((b, a) => a[0] - b[0]).slice(0, numOfLines);
        linesR = linesR.sort((b, a) => a[0] - b[0]).slice(0, numOfLines);

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
          }
        }
        for (let i = 0; i < linesR.length; i += 1) {
          let line = new gm.Line();
          line.fromParallelCoords(
            linesR[i][1] * 2,
            linesR[i][2] * 2,
            canvasR.width,
            canvasR.height,
            maxP,
            maxP / 2
          );
          if (Math.abs(line.angle - 90) > 89) {
            // 90
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
        gm.canvasDrawLine(canvas, horizontal[0], "rgba(0, 255, 0, 1.0)");

        setGapH(Math.abs(horizontal[0].y1 + horizontal[0].y2 - canvas.height));
        console.log(
          gapH,
          Math.abs(horizontal[0].y1 + horizontal[0].y2 - canvas.height)
        );

        for (let line of vertical) {
          gm.canvasDrawLine(canvasR, line, "rgba(0, 0, 255, 1.0)");
        }

        // use ckmeans to find the natural breaks
        const xCorArray = vertical.map((line) => (line.y1 + line.y2) / 2);
        let finalBreaks;
        for (let i = 2; i < xCorArray.length; i++) {
          const breaks = ckmeans(xCorArray, i);
          let vari = 0;
          for (let cluster of breaks) {
            vari += variance(cluster);
          }
          // unsupervised
          if (vari < 20) {
            finalBreaks = breaks;
            break;
          }
        }
        console.log("finalBreaks", finalBreaks);

        // calc and sort all x cors
        const xCors = finalBreaks
          .map((cluster) => mean(cluster))
          .sort((a, b) => a - b);

        // calc difference of them
        const diffs = [];
        for (let i = 0; i < xCors.length - 1; i++) {
          diffs.push(xCors[i + 1] - xCors[i]);
        }
        console.log("diffs", diffs);

        // use ckmeans again to find the natural breaks
        let diffBreaks;
        for (let i = 2; i < diffs.length; i++) {
          const breaks = ckmeans(diffs, i);
          let vari = 0;
          for (let cluster of breaks) {
            vari += variance(cluster);
          }
          // unsupervised
          if (vari < 20) {
            diffBreaks = breaks;
            break;
          }
        }
        console.log(diffBreaks);
        const gapW = mode(diffs);
        console.log("gap", gapW);

        // for (let line of vertical) {
        //   gm.canvasDrawLine(canvas, line, "rgba(0, 0, 255, 1.0)");
        // }
      })();
    }
  }, [hough, ori, low, high]);

  return (
    <div>
      <input
        className="form-control mb-3"
        type="file"
        onChange={handleUpload}
      />
      <div className="d-flex">
        <label>
          Canny Low
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={low}
            onChange={(evt) => setLow(parseFloat(evt.target.value))}
          />
        </label>
        <label>
          Canny High
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={high}
            onChange={(evt) => setHigh(parseFloat(evt.target.value))}
          />
        </label>
      </div>
      <div className="mb-3">
        <button
          className="btn btn-outline-primary me-3"
          onClick={() => setHough(true)}
        >
          Start
        </button>
        <button className="btn btn-outline-primary" onClick={reset}>
          Reset
        </button>
      </div>
      <canvas />
      <canvas />
    </div>
  );
}
