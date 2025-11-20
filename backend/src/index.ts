import express, { type Express, type Request, type Response } from "express";

const app: Express = express();
const port = process.env.PORT || 3000;

app.get("/", (req: Request, res: Response) => {
  res.send("Hello World with TypeScript and Express!");
});

app.listen(port, () => {
  console.log(`Backend server is running at http://localhost:${port}`);
});
