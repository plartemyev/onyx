"use client";

import { ReactNode, useState } from "react";
import { cn } from "@opal/utils";
import { ChatFileType, FileDescriptor } from "@/app/app/interfaces";
import Attachment from "@/refresh-components/Attachment";
import { InMessageImage } from "@/app/app/components/files/images/InMessageImage";
import InMessageMedia from "@/app/app/components/files/media/InMessageMedia";
import { resolveChatFileType } from "@/app/app/components/files/media/utils";
import SpreadsheetContent from "@/components/tools/SpreadsheetContent";
import PreviewModal from "@/sections/modals/PreviewModal";
import { MinimalOnyxDocument } from "@/lib/search/interfaces";
import ExpandableContentWrapper from "@/components/tools/ExpandableContentWrapper";
import { useProjectsContextOptional } from "@/lib/projects/providers";
import { UserFileStatus } from "@/lib/projects/types";

// Statuses in which the file's background processing (extraction, indexing,
// captioning) has not finished — the chip shows a live pending state.
const PENDING_FILE_STATUSES = new Set([
  UserFileStatus.UPLOADED,
  UserFileStatus.PROCESSING,
  UserFileStatus.INDEXING,
]);

interface FileContainerProps {
  children: ReactNode;
  className?: string;
  id?: string;
  align: "start" | "end";
}

interface FileDisplayProps {
  files: FileDescriptor[];
  /** Cross-axis alignment of each file row. Defaults to "end" (user bubbles). */
  align?: "start" | "end";
}

function FileContainer({ children, className, id, align }: FileContainerProps) {
  return (
    <div
      id={id}
      className={cn(
        "flex w-full flex-col gap-2 py-2",
        align === "end" ? "items-end" : "items-start",
        className
      )}
    >
      {children}
    </div>
  );
}

export default function FileDisplay({
  files,
  align = "end",
}: FileDisplayProps) {
  const [close, setClose] = useState(true);
  const [previewingFile, setPreviewingFile] = useState<FileDescriptor | null>(
    null
  );
  // Live processing status for files uploaded in this browser session (fed by
  // the ProjectsProvider status poll). Files from history before a reload
  // have no entry and render statically. Optional: FileDisplay also mounts in
  // the Craft flow, outside the provider.
  const projectsContext = useProjectsContextOptional();
  const userFileStatusById = projectsContext?.userFileStatusById ?? {};
  const isProcessing = (file: FileDescriptor): boolean => {
    const userFileId = file.user_file_id;
    if (!userFileId) return false;
    const status = userFileStatusById[userFileId];
    return status !== undefined && PENDING_FILE_STATUSES.has(status);
  };
  const textFiles = files.filter(
    (file) =>
      resolveChatFileType(file) === ChatFileType.PLAIN_TEXT ||
      resolveChatFileType(file) === ChatFileType.DOCUMENT
  );
  const imageFiles = files.filter(
    (file) => resolveChatFileType(file) === ChatFileType.IMAGE
  );
  const tabularFiles = files.filter(
    (file) => resolveChatFileType(file) === ChatFileType.TABULAR
  );
  const mediaFiles = files.filter(
    (file) =>
      resolveChatFileType(file) === ChatFileType.AUDIO ||
      resolveChatFileType(file) === ChatFileType.VIDEO
  );

  const presentingDocument: MinimalOnyxDocument = {
    document_id: previewingFile?.id ?? "",
    semantic_identifier: previewingFile?.name ?? "",
  };

  return (
    <>
      {previewingFile && (
        <PreviewModal
          presentingDocument={presentingDocument}
          onClose={() => setPreviewingFile(null)}
        />
      )}

      {textFiles.length > 0 && (
        <FileContainer id="onyx-file" align={align}>
          {textFiles.map((file) => (
            <Attachment
              key={file.id}
              fileName={file.name || file.id}
              open={() => setPreviewingFile(file)}
              processing={isProcessing(file)}
            />
          ))}
        </FileContainer>
      )}

      {imageFiles.length > 0 && (
        <FileContainer id="onyx-image" align={align}>
          {imageFiles.map((file) => (
            <InMessageImage key={file.id} fileId={file.id} />
          ))}
        </FileContainer>
      )}

      {mediaFiles.length > 0 && (
        <FileContainer align={align}>
          {mediaFiles.map((file) => (
            <InMessageMedia
              key={file.id}
              fileId={file.id}
              fileName={file.name ?? undefined}
              kind={
                resolveChatFileType(file) === ChatFileType.AUDIO
                  ? "audio"
                  : "video"
              }
            />
          ))}
        </FileContainer>
      )}

      {tabularFiles.length > 0 && (
        <FileContainer className="overflow-auto" align={align}>
          {tabularFiles.map((file) =>
            close ? (
              <ExpandableContentWrapper
                key={file.id}
                fileDescriptor={file}
                close={() => setClose(false)}
                // All TABULAR files render via the parsed preview, which
                // dispatches on the response content type: spreadsheet JSON
                // (xlsx is a binary OOXML zip the browser can't decode as
                // text) or raw CSV passthrough. Display names are never
                // trusted — a renamed xlsx/csv still renders correctly.
                ContentComponent={SpreadsheetContent}
              />
            ) : (
              <Attachment
                key={file.id}
                open={() => setClose(true)}
                fileName={file.name || file.id}
                processing={isProcessing(file)}
              />
            )
          )}
        </FileContainer>
      )}
    </>
  );
}
