import "bootstrap/dist/css/bootstrap.min.css";
import React, { useState, useEffect } from "react";
import jsfeat from "jsfeat";

export default function TeamRecog() {
  const [originalImage, setOri] = useState();

  const [canny, setCanny] = useState(false);
  const [blur, setBlur] = useState(1);
  const [lowThreshold, setLow] = useState(30);
  const [highThreshold, setHigh] = useState(100);

  const [isHough, setHough] = useState(false);

  function handleUpload(evt) {
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const reader = new FileReader();
    reader.readAsDataURL(evt.target.files[0]);
    reader.onload = (evt) => {
      const image = new Image();
      image.onload = () => {
        canvas.width = image.width;
        canvas.height = image.height;
        ctx.drawImage(image, 0, 0);
      };
      image.src = evt.target.result.toString();
      setOri(image);
    };
  }

  useEffect(() => {
    if (canny && !isHough) {
      const canvas = document.querySelector("canvas");
      const ctx = canvas.getContext("2d");

      const width = canvas.width;
      const height = canvas.height;

      const data_type = jsfeat.U8_t | jsfeat.C1_t;
      const img = originalImage;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      const data_buffer = new jsfeat.data_t(width * height);
      const img_u8 = new jsfeat.matrix_t(width, height, data_type, data_buffer);
      const imageData = ctx.getImageData(0, 0, width, height);

      // to greyscale
      jsfeat.imgproc.grayscale(imageData.data, width, height, img_u8);

      // blur
      const r = blur | 0;
      const kernel_size = (r + 1) << 1;
      jsfeat.imgproc.gaussian_blur(img_u8, img_u8, kernel_size, 0);

      // canny
      jsfeat.imgproc.canny(img_u8, img_u8, lowThreshold | 0, highThreshold | 0);

      // render
      const data_u32 = new Uint32Array(imageData.data.buffer);
      const alpha = 0xff << 24;
      let i = img_u8.cols * img_u8.rows,
        pix = 0;
      while (--i >= 0) {
        pix = img_u8.data[i];
        data_u32[i] = alpha | (pix << 16) | (pix << 8) | pix;
      }

      // draw
      ctx.putImageData(imageData, 0, 0);
    }
  }, [canny, blur, lowThreshold, highThreshold]);

  return (
    <div>
      <input
        className="form-control mb-3"
        type="file"
        onChange={handleUpload}
      />
      {originalImage ? (
        <div className="mb-3">
          <button
            className="btn btn-outline-primary me-3"
            onClick={() => setCanny(true)}
          >
            Canny
          </button>
          {canny ? (
            <button
              className="btn btn-outline-primary me-3"
              onClick={() => setHough(true)}
            >
              Hough
            </button>
          ) : null}
        </div>
      ) : null}
      {canny ? (
        <div className="d-flex mb-3">
          <input
            className="form-control me-3"
            type="number"
            value={blur}
            onChange={(evt) => setBlur(evt.target.value)}
          />
          <input
            className="form-control me-3"
            type="number"
            value={lowThreshold}
            onChange={(evt) => setLow(evt.target.value)}
          />
          <input
            className="form-control me-3"
            type="number"
            value={highThreshold}
            onChange={(evt) => setHigh(evt.target.value)}
          />
        </div>
      ) : null}
      <canvas />
      <canvas id="hough" />
    </div>
  );
}
