import Image from "next/image";

export default function TransactionsIcon({
  height = 16,
  width = height,
  margin = "0px",
}: {
  height?: number;
  width?: number;
  margin?: string;
}) {
  return (
    <Image
      src="/icons/arrows-icon.svg"
      alt="Transactions Icon"
      height={height}
      width={height}
      style={{ margin: margin }}
    ></Image>
  );
}
