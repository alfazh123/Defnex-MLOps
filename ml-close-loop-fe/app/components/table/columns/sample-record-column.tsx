"use client"

import { createColumnHelper } from "@tanstack/react-table"

import type { RecordSample } from "~/type"
import type { DataTableFeatures } from "../table-features"

// Use `accessor` for data columns and `display` for columns without one.
const columnHelper = createColumnHelper<DataTableFeatures, RecordSample>()

export const columnsRecordSample = columnHelper.columns([
    columnHelper.accessor("id", {
        header: "#",
    }),
    columnHelper.accessor("screnarioId", {
        header: "Scenario ID",
    }),
    columnHelper.accessor("domain", {
        header: "Domain",
    }),
    columnHelper.accessor("environment", {
        header: "Environment",
    }),
    columnHelper.accessor("time", {
        header: "Time",
    }),
    columnHelper.accessor("location", {
        header: "Location",
    }),
])