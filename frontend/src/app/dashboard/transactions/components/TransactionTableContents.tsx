"use client";

import { Table, Input } from "antd";
import { useTable, FilterDropdown } from "@refinedev/antd";

export interface Transaction {
  txFraud: boolean;
  txFraudScenario: number;
  transactionId: number;
  txDatetime: Date;
  customerId: number;
  terminalId: number;
  txAmount: number;
  txTimeSeconds: number;
  txTimeDays: number;
}

export const TransactionsTableContents: React.FC = () => {
  const { tableProps } = useTable<Transaction>({ resource: "transactions" });

  return (
    <Table {...tableProps} rowKey="transactionId">
      <Table.Column
        dataIndex="txFraud"
        title="Fraud"
        sorter={{ multiple: 1 }}
        render={(value: boolean) => (
          <span style={{ color: value ? "red" : "green", fontWeight: 600 }}>
            {value ? "Fraud" : "Legit"}
          </span>
        )}
      />
      <Table.Column dataIndex="txFraudScenario" title="Fraud Scenario" />
      <Table.Column dataIndex="transactionId" title="Transaction ID" />
      <Table.Column
        dataIndex="txDatetime"
        title="Date/Time"
        render={(value: Date) => new Date(value).toLocaleString()}
        sorter={{ multiple: 2 }}
      />
      <Table.Column
        dataIndex="customerId"
        title="Customer ID"
        filterDropdown={(props) => (
          <FilterDropdown {...props}>
            <Input placeholder="Search by ID" type="number" />
          </FilterDropdown>
        )}
      />
      <Table.Column dataIndex="terminalId" title="Terminal ID" />
      <Table.Column
        dataIndex="txAmount"
        title="Amount"
        sorter={{ multiple: 3 }}
      />
      <Table.Column
        dataIndex="txTimeSeconds"
        title="Time (seconds)"
        sorter={{ multiple: 4 }}
      />
      <Table.Column
        dataIndex="txTimeDays"
        title="Time (days)"
        sorter={{ multiple: 5 }}
      />
    </Table>
  );
};
