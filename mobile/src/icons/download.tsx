import Svg, { Path } from "react-native-svg";

import type { IconProps } from "@/icons/types";

// Port of web Opal `download`.
const SvgDownload = ({ size = 16, ...props }: IconProps) => (
  <Svg
    width={size}
    height={size}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    {...props}
  >
    <Path
      d="M14 10V12.6667C14 13.3929 13.3929 14 12.6667 14H3.33333C2.60711 14 2 13.3929 2 12.6667V10M4.66667 6.66667L8 10M8 10L11.3333 6.66667M8 10L8 2"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </Svg>
);

export default SvgDownload;
