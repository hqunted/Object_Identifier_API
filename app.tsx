const crypto = require("crypto");
const express = require("express");
const multer = require("multer");
const cors = require("cors");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const app = express();
const path = require("path");
const port = 5000;
const allowedOrigins = ["http://localhost:3000"];
const mobilenet = require("@tensorflow-models/mobilenet");
const Jimp = require("jimp");
const { createCanvas, loadImage } = require("canvas");
const http = require("http").Server(app);
const io = require("socket.io")(http);

app.use(
  cors({
    origin: function (origin, callback) {
      if (!origin) {
        return callback(null, true);
      }

      if (allowedOrigins.indexOf(origin) === -1) {
        const msg =
          "The CORS policy for this site does not allow access from the specified Origin.";
        return callback(new Error(msg), false);
      }

      return callback(null, true);
    },
  })
);

//File Deleter
const deleteFile = (filePath) => {
  fs.unlink(filePath, (err) => {
    if (err) {
      console.error(err);
    } else {
      console.log(`Deleted file ${filePath}`);
    }
  });
};
//Base64 converter
const getBase64Image = (filePath) => {
  const imageData = fs.readFileSync(filePath);
  const base64Image = imageData.toString("base64");
  return base64Image;
};

//Object Identification
async function predict(imagePath) {
  // load the COCO-SSD model
  const model = await cocoSsd.load();

  // load the image to be detected
  const img = await Jimp.read(imagePath);
  const buffer = await img.getBufferAsync(Jimp.MIME_JPEG);
  const tfImg = tf.node.decodeJpeg(buffer);

  // predict using the COCO-SSD model
  const predictions = await model.detect(tfImg);

  // draw bbox on image
  const canvas = createCanvas(img.bitmap.width, img.bitmap.height);
  const ctx = canvas.getContext("2d");
  const imgData = img.bitmap.data;
  const imageData = ctx.createImageData(img.bitmap.width, img.bitmap.height);

  imageData.data.set(imgData);
  ctx.putImageData(imageData, 0, 0);
  ctx.strokeStyle = "#00FF00";
  ctx.lineWidth = 3;
  ctx.font = "32px Sans-serif";

  predictions.forEach((pred) => {
    const [x, y, w, h] = pred.bbox;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#00FF00";
    ctx.fillText(pred.class, x, y - 5);
  });

  // convert canvas to Jimp image
  const newImg = await Jimp.read(canvas.toBuffer("image/jpeg"));

  // save image with bbox to disk
  const filenameWithBbox = `${crypto.randomUUID()}_bbox.jpg`;
  sendDataToClient(`./uploads/${filenameWithBbox}`);
  fs.unlink(filenameWithBbox, (err) => {
    if (err) {
      console.error(err);
    } else {
      console.log(`Deleted file ${filenameWithBbox}`);
    }
  });

  await newImg.writeAsync(`./uploads/${filenameWithBbox}`);
  deleteFile(imagePath);
  return predictions;
}

const sendDataToClient = (imageData) => {
  io.on("connection", (socket) => {

    // Send image data to client
    socket.emit("imageData", getBase64Image(imageData));

    //Delete imageData after sending
    deleteFile(imageData);

    // Disconnect the client after sending the data
    socket.disconnect();
  });

  http.listen(3000, () => {
    console.log("Server started on port 3000");
  });
};

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const filename = `${crypto.randomUUID()}.${file.originalname}`;
    cb(null, filename);
    //Making prediction

    predict(`./uploads/${filename}`).then((predictions) => {});
  },
});
const upload = multer({ storage });

//  Communication
app.post("/api/upload-image", upload.single("image"), (req, res) => {
  res.send("File uploaded successfully!");
});

app.get("/", (req, res) => {});

app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
