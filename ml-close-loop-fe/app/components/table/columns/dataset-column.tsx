"use client"

import { createColumnHelper } from "@tanstack/react-table"

import { type DataTableFeatures } from "../dataset-table-features"
import type { Dataset } from "~/type"
import clsx from "clsx"

// Use `accessor` for data columns and `display` for columns without one.
const columnHelper = createColumnHelper<DataTableFeatures, Dataset>()

export const columns = columnHelper.columns([
    columnHelper.accessor("id", {
        header: "ID",
    }),
    columnHelper.accessor("title", {
        header: "Title",
    }),
    // columnHelper.accessor("fileName", {
    //     header: "File Name",
    // }),
    columnHelper.accessor("format", {
        header: "Format",
    }),
    columnHelper.accessor("category", {
        header: "Category",
    }),
    columnHelper.accessor("fileSize", {
        header: "File Size",
        cell: ({row}) => {
            const fileSize = row.original.fileSize.toFixed(1);
            return `${fileSize} MB`;
        }
    }),
    columnHelper.accessor("totalRows", {
        header: "Total Rows",
    }),
    columnHelper.accessor("validationStatus", {
        header: () => (
            <div className="flex justify-center items-center">
                Validation Status
            </div>
        ),
        cell: ({row}) => {
            return (
                <div className={`rounded-md text-sm font-medium flex justify-center items-center`} >
                    <span className={clsx(
                        "px-2 py-1 rounded-md text-sm font-medium",
                        row.original.validationStatus === "Valid" && "bg-green-100 text-green-800",
                        row.original.validationStatus === "Invalid" && "bg-yellow-100 text-yellow-800",
                        row.original.validationStatus === "Error" && "bg-red-100 text-red-800"
                    )}>
                        {row.original.validationStatus}
                    </span>
                </div>
            )
        }
    }),
    columnHelper.accessor("sftStatus", {
        header: () => (
            <div className="flex justify-center items-center">
                SFT Status
            </div>
        ),
        cell: ({row}) => {
            return (
                <div className={`rounded-md text-sm font-medium flex justify-center items-center`} >
                    <span className={clsx(
                        "px-2 py-1 rounded-md text-sm font-medium",
                        row.original.sftStatus === "Ready" && "bg-green-100 text-green-800",
                        row.original.sftStatus === "Training" && "bg-blue-100 text-blue-800",
                        row.original.sftStatus === "Queued" && "bg-yellow-100 text-yellow-800",
                        row.original.sftStatus === "Failed" && "bg-red-100 text-red-800"
                    )}>
                        {row.original.sftStatus}
                    </span>
                </div>
            )
        }
    }),
    columnHelper.accessor("uploadedBy", {
        header: "Uploaded By",
    }),
    columnHelper.accessor("uploadedAt", {
        header: "Uploaded At",
        cell: ({row}) => {
            const uploadedAt = new Date(row.original.uploadedAt);
            const formattedDate = uploadedAt.toLocaleString("en-US", {
                year: "numeric",
                month: "short",
                day: "numeric",
                // hour: "2-digit",
                // minute: "2-digit",
            });
            return formattedDate;
        }
    }),
])