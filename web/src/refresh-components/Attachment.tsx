import Text from "@/refresh-components/texts/Text";
import { Button } from "@opal/components";
import { SvgFileText, SvgMaximize2, SvgSimpleLoader } from "@opal/icons";
import { useTranslations } from "next-intl";
export interface AttachmentsProps {
  fileName: string;
  open?: () => void;
  /** Background processing (indexing/captioning) is still running for this
   * file — show a spinner instead of the static file icon. */
  processing?: boolean;
}

export default function Attachments({
  fileName,
  open,
  processing = false,
}: AttachmentsProps) {
  const t = useTranslations("common.attachment");
  return (
    <div className="flex items-center border bg-background-tint-00 rounded-12 p-1 gap-1">
      <div className="p-2 bg-background-tint-01 rounded-08">
        {processing ? (
          <SvgSimpleLoader className="w-5 h-5 stroke-text-02" />
        ) : (
          <SvgFileText className="w-5 h-5 stroke-text-02" />
        )}
      </div>
      <div className="flex flex-col px-2">
        <Text as="p" secondaryAction>
          {fileName}
        </Text>
        <Text as="p" secondaryBody text03>
          {processing ? t("processing.description") : t("document.label")}
        </Text>
      </div>

      {open && (
        <Button
          aria-label={t("expandButton.ariaLabel")}
          onClick={open}
          icon={SvgMaximize2}
          prominence="tertiary"
          size="sm"
        />
      )}
    </div>
  );
}
