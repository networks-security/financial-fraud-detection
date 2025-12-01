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
    const downloadProcess = spawn('python', [downloadScript]);

    downloadProcess.stdout.on('data', (data) => {
        console.log(`Python stdout: ${data}`);
    });

    downloadProcess.stderr.on('data', (data) => {
        console.error(`Python stderr: ${data}`);
    });

    downloadProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
    });
}