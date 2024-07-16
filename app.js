const recordButton = document.getElementById("recordButton")
const stopButton = document.getElementById("stopButton")

let mediaRecorder
let audioChunks = []

recordButton.addEventListener("click", async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder = new MediaRecorder(stream)

    mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data)
    }

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" })
        const formData = new FormData()
        formData.append("file", audioBlob, "audio.wav")

        const response = await fetch("http://localhost:5000/transcribe", {
            method: "POST",
            body: formData,
        })

        const result = await response.json()
        console.log(result)
    }

    mediaRecorder.start()
    recordButton.disabled = true
    stopButton.disabled = false
})

stopButton.addEventListener("click", () => {
    mediaRecorder.stop()
    recordButton.disabled = false
    stopButton.disabled = true
})
