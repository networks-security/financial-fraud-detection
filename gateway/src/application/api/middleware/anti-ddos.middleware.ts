import { type Request, type Response, type NextFunction } from "express";

// Simple in-memory store: { ip: { count, firstRequestTime } }
const requestCounts: Record<
  string,
  { count: number; firstRequestTime: number }
> = {};

const WINDOW_SIZE = 60 * 1000; // 1 minute in ms
const MAX_REQUESTS = 4; // max requests per IP per window

export function ddosMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
) {
  // Ensure IP is defined, fallback to 'unknown'
  const ip = req.ip || "unknown";

  const now = Date.now();
  const record = requestCounts[ip];

  if (!record) {
    requestCounts[ip] = { count: 1, firstRequestTime: now };
    return next();
  }

  // Check if current window has expired
  if (now - record.firstRequestTime > WINDOW_SIZE) {
    requestCounts[ip] = { count: 1, firstRequestTime: now };
    return next();
  }

  // Increment request count
  record.count += 1;

  if (record.count > MAX_REQUESTS) {
    // Too many requests
    console.warn(`DDoS protection: IP ${ip} blocked for too many requests.`);
    return res
      .status(429)
      .json({ error: "Too many requests, try again later." });
  }

  next();
}
