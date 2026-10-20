import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react-vite";
import { ModelSettingsPopover } from "@/sections/modals/languageModels/ModelSettingsPopover";
import type { ModelConfiguration } from "@/lib/languageModels/types";

const meta = {
  title: "Sections/languageModels/ModelSettingsPopover",
  component: ModelSettingsPopover,
} satisfies Meta<typeof ModelSettingsPopover>;

export default meta;

type Story = StoryObj<typeof meta>;

const noop = () => {};
const baseModel: ModelConfiguration = {
  name: "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
  is_visible: true,
  max_input_tokens: 262144,
  supports_image_input: false,
  supports_reasoning: false,
  effectiveDisplayName: "Qwen3.6 35B A3B",
};

/** Controlled harness mirroring the app: patches flow back into the model. */
function PopoverHarness({
  modelOverrides = {},
}: {
  modelOverrides?: Partial<ModelConfiguration>;
}) {
  const [model, setModel] = useState<ModelConfiguration>({
    ...baseModel,
    ...modelOverrides,
  });
  return (
    <div className="p-4">
      <ModelSettingsPopover
        model={model}
        onChange={(patch) => setModel((m) => ({ ...m, ...patch }))}
      />
    </div>
  );
}

export const VisionCapable: Story = {
  args: { model: baseModel, onChange: noop },
  render: () => (
    <PopoverHarness
      modelOverrides={{
        supports_image_input: true,
        temperature_default: 0.7,
      }}
    />
  ),
};

export const ReasoningModel: Story = {
  args: { model: baseModel, onChange: noop },
  render: () => (
    <PopoverHarness
      modelOverrides={{
        supports_reasoning: true,
        supported_reasoning_efforts: ["low", "medium", "high"],
      }}
    />
  ),
};
