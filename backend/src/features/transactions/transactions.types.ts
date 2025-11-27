export interface Transaction {
  transactionId: number;
  txDatetime: Date;
  customerId: number;
  terminalId: number;
  txAmount: number;
  txTimeSeconds: number;
  txTimeDays: number;
}
