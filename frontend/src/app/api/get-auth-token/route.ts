import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const object = req.cookies.get("firebaseToken")?.value;

  if (!object) {
    return NextResponse.json({ error: "No token found" }, { status: 404 });
  }

  const token = JSON.parse(object)["token"];
  console.log("Found a saved token in cookies, returning", token);
  return NextResponse.json(token);
}
