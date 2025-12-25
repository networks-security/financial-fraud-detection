import { Router } from "express";
import { getTransactions } from "./dashboard.controller.ts";

const dashboardRoutes = Router();
dashboardRoutes.get("/transactions", getTransactions);

export default dashboardRoutes;
