import "./globals.scss";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <title>Financial Fraud Detection</title>
      </head>
      <body>{children}</body>
    </html>
  );
}
