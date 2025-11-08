import Image from "next/image";

export default function AnalysisIcon({
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
      src="/icons/analysis-icon.svg"
      alt="Transactions Icon"
      height={height}
      width={height}
      style={{ margin: margin }}
    ></Image>
  );
}
