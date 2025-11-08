import Image from "next/image";

export default function AccountDetailsIcon({
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
      src="/icons/person-icon.svg"
      alt="Transactions Icon"
      height={height}
      width={height}
      style={{ margin: margin }}
    ></Image>
  );
}
