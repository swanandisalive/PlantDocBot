function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const descriptionInput = document.getElementById('plantDescription');
    const file = fileInput.files[0];
    
    const resultDiv = document.getElementById('result');
    const imagePreview = document.getElementById('imagePreview');

    if (!file) {
        alert("Please select an image first!");
        return;
    }

    // 1. Show Preview
    imagePreview.src = URL.createObjectURL(file);
    imagePreview.style.display = "block";
    resultDiv.style.display = "block";
    resultDiv.innerHTML = "<p style='color:#666;'>🔍 Analyzing image and text...</p>";

    const formData = new FormData();
    formData.append('file', file);
    formData.append('description', descriptionInput.value); 

    // 2. Send to Backend
    fetch('http://127.0.0.1:5000/api/analyze', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error("Server error");
        return response.json();
    })
    .then(data => {
        console.log("Data:", data);

        // --- 3. DISPLAY RESULTS ---
        
        // Colors
        const imgColor = data.image.status === "Healthy" ? "#27ae60" : "#c0392b";
        const textColor = data.text.diagnosis === "Healthy" ? "#27ae60" : "#c0392b";
        
        // Check if text was actually analyzed
        let textHTML = "";
        if (data.text.diagnosis !== "Not Provided") {
            textHTML = `
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #eee;">
                    <h3 style="margin: 0; color: #333;">📝 Symptom Analysis</h3>
                    <p>Based on your description: <strong style="color: ${textColor}">${data.text.diagnosis}</strong></p>
                    <small>Confidence: ${data.text.confidence}</small>
                </div>
            `;
        } else {
            textHTML = `<p style="color: #999; font-style:italic; margin-top:10px;">(No description provided)</p>`;
        }

        resultDiv.innerHTML = `
            <div>
                <h3 style="margin: 0; color: #333;">📸 Visual Analysis</h3>
                <p><strong>Prediction:</strong> <span style="color: ${imgColor}">${data.image.diagnosis}</span></p>
                <small>Confidence: ${data.image.confidence}</small>
            </div>

            ${textHTML}
        `;
    })
    .catch(error => {
        console.error(error);
        resultDiv.innerHTML = "<p style='color:red'>❌ Error connecting to server.</p>";
    });
}