import { useState } from "react";
import {
  render,
  screen,
  setupUser,
} from "@tests/setup/test-utils";
import { ModelSettingsPopover } from "@/sections/modals/languageModels/ModelSettingsPopover";
import type { ModelSettingsPatch } from "@/sections/modals/languageModels/ModelSettingsPopover";
import type { ModelConfiguration } from "@/lib/languageModels/types";

/** Minimal fixture satisfying the popover's ModelSettingsModel pick. */
function makeModel(
  overrides: Partial<ModelConfiguration> = {}
): ModelConfiguration {
  return {
    name: "qwen3:32b",
    is_visible: true,
    max_input_tokens: 262144,
    supports_image_input: false,
    supports_reasoning: false,
    effectiveDisplayName: "Qwen3 32B",
    ...overrides,
  };
}

/** The popover is controlled by its parent (formik in the app), so mirror
 *  that here: patches must flow back into the model for the input to update. */
function PopoverHarness({
  initialMaxInputTokens,
  onChange,
}: {
  initialMaxInputTokens: number | null;
  onChange?: (patch: ModelSettingsPatch) => void;
}) {
  const [model, setModel] = useState<ModelConfiguration>(
    makeModel({ max_input_tokens: initialMaxInputTokens })
  );
  return (
    <ModelSettingsPopover
      model={model}
      onChange={(patch) => {
        onChange?.(patch);
        setModel((m) => ({ ...m, ...patch }));
      }}
    />
  );
}

async function openPopover() {
  const user = setupUser();
  await user.click(screen.getByRole("button"));
  return user;
}

describe("ModelSettingsPopover context window", () => {
  it("shows the stored context window", async () => {
    render(<PopoverHarness initialMaxInputTokens={262144} />);
    await openPopover();
    expect(
      screen.getByRole("spinbutton", { name: "Context Window" })
    ).toHaveValue(262144);
  });

  it("shows no value when the context window is unset", async () => {
    render(<PopoverHarness initialMaxInputTokens={null} />);
    await openPopover();
    expect(
      screen.getByRole("spinbutton", { name: "Context Window" })
    ).toHaveValue(null);
  });

  it("patches the context window on edit", async () => {
    const onChange = jest.fn();
    render(
      <PopoverHarness initialMaxInputTokens={262144} onChange={onChange} />
    );
    const user = await openPopover();
    const input = screen.getByRole("spinbutton", { name: "Context Window" });
    await user.clear(input);
    await user.type(input, "16384");
    expect(onChange).toHaveBeenLastCalledWith({ max_input_tokens: 16384 });
    expect(input).toHaveValue(16384);
  });

  it("clears the override so the provider default applies", async () => {
    const onChange = jest.fn();
    render(<PopoverHarness initialMaxInputTokens={262144} onChange={onChange} />);
    const user = await openPopover();
    await user.clear(screen.getByRole("spinbutton", { name: "Context Window" }));
    expect(onChange).toHaveBeenLastCalledWith({ max_input_tokens: null });
  });
});
