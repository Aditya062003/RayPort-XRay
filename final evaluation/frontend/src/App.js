import React, { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [textInput, setTextInput] = useState("");
  const [imagePreview, setImagePreview] = useState(null);
  const [pdfContent, setPdfContent] = useState(null);
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    const previewURL = URL.createObjectURL(selectedFile);
    setImagePreview(previewURL);
  };

  const handleSubmit = () => {
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("text", textInput);
      axios
        .post("http://127.0.0.1:5000/predict", formData)
        .then((response) => {
          console.log(response.data);
          setPrediction(response.data.prediction);

          setPdfContent(response.data.pdfcontent);
        })
        .catch((error) => console.error("Error:", error));
    } else if (textInput.trim() !== "") {
      console.log("User typed:", textInput);
      setTextInput("");
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col relative">
      {/* Navbar */}
      <nav className="bg-blue-500 p-4 text-white">
        <h1 className="text-2xl font-bold">RayPort</h1>
      </nav>

      <div className="flex mx-8 my-4 justify-center">
        <div className="w-1/4">
          {imagePreview && (
            <div className="flex-shrink-0 mx-auto">
              <img
                src={imagePreview}
                alt="Preview"
                className="mb-4 w-60 h-60 rounded-md shadow"
                style={{ maxWidth: "100%", maxHeight: "600px" }}
              />
            </div>
          )}
        </div>
        <div className="w-1/4">
          {prediction && (
            <div className="mb-8">
              <h2 className="text-xl font-bold mb-2">Prediction:</h2>
              <div className="">
                <table className="table-auto">
                  <thead>
                    <tr>
                      <th className="px-4 py-2">Condition</th>
                      <th className="px-4 py-2">Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prediction.map((prob, index) => (
                      <tr key={index}>
                        <td className="border px-4 py-2">
                          {conditionLabels[index]}
                        </td>
                        <td className="border px-4 py-2">{prob}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
        <div className="w-2/4">
          {pdfContent && (
            <div className="">
              <iframe
                title="Generated PDF"
                src={`data:application/pdf;base64,${pdfContent}`}
                height="700px"
                width="100%"
              ></iframe>
            </div>
          )}
        </div>
      </div>

      <div className="fixed bottom-0 left-0 right-0 p-4 bg-white flex">
        {/* Image Upload Section */}
        <div className="mb-2">
          <input
            type="file"
            id="file"
            className="mt-1 p-2 border rounded-md"
            onChange={handleFileChange}
            accept="image/*"
          />
        </div>

        {/* Chat Input Section */}
        <div className="ml-4 mx-4 px-4 w-full">
          <input
            type="text"
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            placeholder="Enter your symptoms here"
            className="border mx-2 mt-1 px-4 py-2 w-full"
          />
        </div>

        {/* Submit Button */}
        <button
          className="bg-blue-500 mx-4 text-white px-4 py-2 rounded"
          onClick={handleSubmit}
        >
          Submit
        </button>
      </div>
    </div>
  );
}

const conditionLabels = [
  "No Finding",
  "Enlarged Cardiomediastinum",
  "Cardiomegaly",
  "Lung Opacity",
  "Lung Lesion",
  "Edema",
  "Consolidation",
  "Pneumonia",
  "Atelectasis",
  "Pneumothorax",
  "Pleural Effusion",
  "Pleural Other",
  "Fracture",
  "Support Devices",
];

export default App;
