import Svg, { Path } from "react-native-svg";

import type { IconProps } from "@/icons/types";

// Port of web Opal `pause-circle`.
const SvgPauseCircle = ({ size = 16, ...props }: IconProps) => (
  <Svg
    width={size}
    height={size}
    viewBox="0 0 15 15"
    fill="none"
    stroke="currentColor"
    {...props}
  >
    <Path
      d="M6.08333 9.41667V5.41667M8.75 9.41667V5.41667M14.0833 7.41667C14.0833 11.0986 11.0986 14.0833 7.41667 14.0833C3.73477 14.0833 0.75 11.0986 0.75 7.41667C0.75 3.73477 3.73477 0.75 7.41667 0.75C11.0986 0.75 14.0833 3.73477 14.0833 7.41667Z"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </Svg>
);

export default SvgPauseCircle;
