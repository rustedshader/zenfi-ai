"use client";

interface FileAttachmentProps {
  file: File;
  onRemove?: () => void;
}

export function FileAttachment({ file, onRemove }: FileAttachmentProps) {
  const { type, name, size } = file;

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith("image/")) return "🖼️";
    if (type.startsWith("text/")) return "📄";
    if (type.includes("pdf")) return "📋";
    if (type.includes("json")) return "📊";
    if (type.includes("csv")) return "📈";
    if (type.includes("xml")) return "🗂️";
    return "📎";
  };

  if (type.startsWith("image/")) {
    return (
      <div className="relative group">
        <img
          className="w-24 h-24 object-cover rounded-lg border dark:border-gray-600 shadow-sm"
          src={URL.createObjectURL(file)}
          alt={name}
        />
        {onRemove && (
          <button
            onClick={onRemove}
            className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs opacity-0 group-hover:opacity-100 transition-opacity"
          >
            ×
          </button>
        )}
        <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 truncate max-w-24">
          {name}
        </div>
        <div className="text-xs text-gray-400 dark:text-gray-500">
          {formatFileSize(size)}
        </div>
      </div>
    );
  }

  return (
    <div className="relative group flex flex-col items-center p-3 bg-gray-50 dark:bg-gray-700 rounded-lg border dark:border-gray-600 max-w-24">
      <div className="text-2xl mb-1">{getFileIcon(type)}</div>
      {onRemove && (
        <button
          onClick={onRemove}
          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs opacity-0 group-hover:opacity-100 transition-opacity"
        >
          ×
        </button>
      )}
      <div className="text-xs text-gray-700 dark:text-gray-300 truncate max-w-20 text-center">
        {name}
      </div>
      <div className="text-xs text-gray-400 dark:text-gray-500">
        {formatFileSize(size)}
      </div>
    </div>
  );
}
