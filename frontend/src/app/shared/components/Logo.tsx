import Image from "next/image";

export function HorizontalLogo() {
  return (
    <Image
      src="/horizontal-logo.svg"
      alt="Logotype Icon"
      height={32}
      width={123.5}
    />
  );
}
