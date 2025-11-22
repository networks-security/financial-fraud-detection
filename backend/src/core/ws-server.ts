import { Server as HTTPServer } from "http";
import { Server, Socket } from "socket.io";

// Socket.IO server instance
let io: Server | null = null;

export function initializeSocketIO(server: HTTPServer): void {
  // Already initialized
  if (io) return;

  io = new Server(server, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"],
    },
  });

  io.on("connection", (socket: Socket) => {
    console.log("WS client connected: ", socket.id);

    socket.on("disconnect", () => {
      console.log("WS client disconnected: ", socket.id);
      socket.disconnect();
    });
  });
}

export function deinitializeSocketIO() {
  // Already deinitialized
  if (!io) return;

  // Disconnect all clients
  io.sockets.sockets.forEach((socket) => socket.disconnect(true));

  io.close(() => console.log("Socket.IO server closed"));

  io = null;
}

export function getIO() {
  if (!io) throw new Error("Socket.IO not initialized!");
  return io;
}
