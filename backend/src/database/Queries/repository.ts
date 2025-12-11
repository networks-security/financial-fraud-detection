import { spawn } from 'child_process';

const uploadScript = 'src/firestoreUploadScript.py';
const downloadScript = 'src/firestoreDownloadScript.py';
const args = ['src/test.json']; 


export function uploadjsonToFirestore() {
    const uploadProcess = spawn('python', [uploadScript, ...args]);
    uploadProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
    });

    uploadProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
    });

    uploadProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
    });
}

export function downloadjsonFromFirestore() {
  return new Promise((resolve, reject) => {
    const downloadProcess = spawn("python3", [downloadScript]);

    let output = "";
    let errorOutput = "";

    downloadProcess.stdout.on("data", (data) => {
      output += data.toString(); // collect output
      console.log("\n\n\nWhat python returned is:\n", output, "\n\n\n\n");
    });

    downloadProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    downloadProcess.on("close", (code) => {
      if (code !== 0) {
        return reject(new Error(`Python process failed: ${errorOutput}`));
      }

      try {
        if (output.trim() === "") {
          return resolve([]);
        }

        // clean if necessary
        const clean = output.trim().replace(/'/g, '"');

        const parsed = JSON.parse(clean);
        resolve(parsed);
      } catch (err) {
        if (err instanceof Error) {
          reject(new Error("Failed to parse Python JSON output: " + err.message));
        }
      }
  });
  });
}
