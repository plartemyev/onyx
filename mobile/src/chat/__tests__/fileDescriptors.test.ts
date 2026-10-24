import { describe, expect, it } from "@jest/globals";

import {
  fileDescriptorToDisplayFile,
  mediaKindOf,
  projectFileToFileDescriptor,
  projectFilesToFileDescriptors,
} from "@/chat/fileDescriptors";
import { UserFileStatus } from "@/chat/contracts/projects";
import { ChatFileType } from "@/chat/interfaces";
import { makeProjectFile } from "@/chat/__tests__/fixtures";

describe("projectFileToFileDescriptor", () => {
  it("remaps the record keys to the send payload shape", () => {
    const file = makeProjectFile({
      id: "user-file-9",
      file_id: "blob-abc",
      name: "report.pdf",
      chat_file_type: ChatFileType.DOCUMENT,
    });

    expect(projectFileToFileDescriptor(file)).toEqual({
      id: "blob-abc", // file_id → id
      type: ChatFileType.DOCUMENT, // chat_file_type → type
      name: "report.pdf",
      user_file_id: "user-file-9", // id → user_file_id
    });
  });

  it("maps a list", () => {
    const files = [
      makeProjectFile({ id: "a", file_id: "fa" }),
      makeProjectFile({ id: "b", file_id: "fb" }),
    ];
    expect(projectFilesToFileDescriptors(files).map((d) => d.id)).toEqual([
      "fa",
      "fb",
    ]);
  });
});

describe("fileDescriptorToDisplayFile", () => {
  it("inverts the mapping and marks the file completed for display", () => {
    const display = fileDescriptorToDisplayFile({
      id: "blob-abc",
      type: ChatFileType.IMAGE,
      name: "pic.png",
      user_file_id: "user-file-9",
    });

    expect(display).toMatchObject({
      id: "user-file-9",
      file_id: "blob-abc", // thumbnail is fetched from /chat/file/{file_id}
      name: "pic.png",
      chat_file_type: ChatFileType.IMAGE,
      status: UserFileStatus.COMPLETED,
    });
  });

  it("falls back for a null name / missing user_file_id", () => {
    const display = fileDescriptorToDisplayFile({
      id: "blob-x",
      type: ChatFileType.DOCUMENT,
      name: null,
    });
    expect(display.name).toBe("File");
    expect(display.id).toBe("blob-x"); // user_file_id ?? id
  });
});

describe("mediaKindOf", () => {
  it("reads explicit AUDIO/VIDEO types", () => {
    expect(mediaKindOf({ id: "a", type: ChatFileType.AUDIO })).toBe("audio");
    expect(mediaKindOf({ id: "v", type: ChatFileType.VIDEO })).toBe("video");
  });

  it("falls back to the filename for legacy plain_text descriptors", () => {
    // Media attached before AUDIO/VIDEO existed was persisted as PLAIN_TEXT.
    expect(
      mediaKindOf({
        id: "w",
        type: ChatFileType.PLAIN_TEXT,
        name: "Teddy_Swims_Lose_Control.wav",
      }),
    ).toBe("audio");
    expect(
      mediaKindOf({
        id: "m",
        type: ChatFileType.PLAIN_TEXT,
        name: "clip.MP4",
      }),
    ).toBe("video");
  });

  it("returns null for non-media files", () => {
    expect(
      mediaKindOf({
        id: "d",
        type: ChatFileType.DOCUMENT,
        name: "report.pdf",
      }),
    ).toBeNull();
    expect(mediaKindOf({ id: "u", type: ChatFileType.PLAIN_TEXT })).toBeNull();
  });
});
