import { ChatFileType, FileDescriptor } from "@/app/app/interfaces";
import { resolveChatFileType } from "@/app/app/components/files/media/utils";
import { resolveVariant } from "@/sections/modals/PreviewModal/variants";

function descriptor(type: ChatFileType, name: string | null): FileDescriptor {
  return { id: "file-id", type, name };
}

describe("resolveChatFileType", () => {
  it("keeps explicit AUDIO/VIDEO types", () => {
    expect(resolveChatFileType(descriptor(ChatFileType.AUDIO, null))).toBe(
      ChatFileType.AUDIO
    );
    expect(resolveChatFileType(descriptor(ChatFileType.VIDEO, null))).toBe(
      ChatFileType.VIDEO
    );
  });

  it("recovers media types from the filename on legacy descriptors", () => {
    // Files attached before AUDIO/VIDEO existed were persisted as PLAIN_TEXT.
    expect(
      resolveChatFileType(
        descriptor(ChatFileType.PLAIN_TEXT, "Lady_Gaga_Die_with_a_Smile.wav")
      )
    ).toBe(ChatFileType.AUDIO);
    expect(
      resolveChatFileType(descriptor(ChatFileType.PLAIN_TEXT, "clip.mp4"))
    ).toBe(ChatFileType.VIDEO);
  });

  it("leaves non-media files untouched", () => {
    expect(
      resolveChatFileType(descriptor(ChatFileType.PLAIN_TEXT, "notes.txt"))
    ).toBe(ChatFileType.PLAIN_TEXT);
    expect(resolveChatFileType(descriptor(ChatFileType.IMAGE, "pic.png"))).toBe(
      ChatFileType.IMAGE
    );
  });
});

describe("PreviewModal media variants", () => {
  it("matches audio by extension and by mime", () => {
    expect(resolveVariant("song.wav", "audio/x-wav").needsRawStream).toBe(true);
    expect(
      resolveVariant("song.wav", "application/octet-stream").needsRawStream
    ).toBe(true);
  });

  it("matches video by extension and by mime", () => {
    expect(resolveVariant("clip.mp4", "video/mp4").needsRawStream).toBe(true);
    expect(
      resolveVariant("clip.mp4", "application/octet-stream").needsRawStream
    ).toBe(true);
    expect(
      resolveVariant("clip.webm", "application/octet-stream").needsRawStream
    ).toBe(true);
  });

  it("does not stream non-media files", () => {
    expect(
      resolveVariant("notes.txt", "application/octet-stream").needsRawStream
    ).toBeFalsy();
    expect(
      resolveVariant("doc.pdf", "application/pdf").needsRawStream
    ).toBeFalsy();
  });
});
