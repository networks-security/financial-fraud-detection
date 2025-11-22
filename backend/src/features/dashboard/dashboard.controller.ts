import { type Transaction } from "./dashboard.types.ts";
import { type Request, type Response } from "express";

export async function getTransactions(_req: Request, res: Response) {
  console.log("getTransactions called");

  // TODO: auth check

  // TODO: fetch for specific userId

  const sampleData: Transaction[] = [
    {
      transactionId: 1,
      txDatetime: new Date("2025-11-08T11:30:00"),
      customerId: 105,
      terminalId: 6,
      txAmount: 30.89,
      txTimeSeconds: 4690,
      txTimeDays: 5,
      txFraud: false,
      txFraudScenario: 0,
    },
    {
      transactionId: 1,
      txDatetime: new Date("2025-11-08T11:30:00"),
      customerId: 105,
      terminalId: 6,
      txAmount: 30.89,
      txTimeSeconds: 4690,
      txTimeDays: 5,
      txFraud: false,
      txFraudScenario: 0,
    },
    {
      transactionId: 2,
      txDatetime: new Date("2025-11-08T11:30:00"),
      customerId: 105,
      terminalId: 6,
      txAmount: 1510.21,
      txTimeSeconds: 4690,
      txTimeDays: 5,
      txFraud: false,
      txFraudScenario: 0,
    },
    {
      transactionId: 3,
      txDatetime: new Date("2025-11-08T11:30:00"),
      customerId: 105,
      terminalId: 6,
      txAmount: 100,
      txTimeSeconds: 4690,
      txTimeDays: 5,
      txFraud: false,
      txFraudScenario: 0,
    },
  ];

  // TODO: add DTOs
  res.json(sampleData);
}
