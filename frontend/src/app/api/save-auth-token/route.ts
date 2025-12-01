import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const token = req.body;

  if (!token) {
    return NextResponse.json({ message: "No token provided" }, { status: 400 });
  }

  const res = NextResponse.json({ success: true });

  // Set HTTP-only secure cookie
  const tokenString = await streamToString(token);
  res.cookies.set("firebaseToken", tokenString);

  return res;
}

async function streamToString(stream: ReadableStream<Uint8Array>) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let result = "";
  let done = false;

  while (!done) {
    const { value, done: readerDone } = await reader.read();
    done = readerDone;
    if (value) {
      result += decoder.decode(value, { stream: !done });
    }
  }

  return result;
}
