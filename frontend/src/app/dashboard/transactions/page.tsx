import QProviders from "./components/QProviders";

import TransactionsTable from "./components/TransactionTable";

export default function Page() {
  return (
    <div>
      <h1>Welcome to the Dashboard Transactions page</h1>
      <p>
        Here&apos;s the list of the most recent transactions in your institution
      </p>
      <QProviders>
        <TransactionsTable />
      </QProviders>
    </div>
  );
}
