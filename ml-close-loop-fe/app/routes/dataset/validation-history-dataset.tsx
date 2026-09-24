import { ArrowLeftIcon, CheckCircleIcon, ClockClockwiseIcon, ClockIcon, ShieldCheckIcon } from "@phosphor-icons/react/dist/ssr";
import { useState } from "react";
import { useParams } from "react-router";
import { Button } from "~/components/ui/button";
import Header from "~/components/ui/header";
import { ScrollArea } from "~/components/ui/scroll-area";
import type { Dataset } from "~/type";

export interface GateCheckItem {
	id: string;
	title: string;
	description: string;
	status: "passed" | "failed";
}

export default function ValidationHistoryDataset({dataset}: {dataset: Dataset}) {
    const params = useParams();
    const slug = params.id;
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<boolean>(false);

    console.log(slug);

    const verificationChecks: GateCheckItem[] = [
		{
			id: "check-1",
			title: "JSONL Syntax & Schema",
			description:
				"Conversations array conforms to ShareGPT/Alpaca role specs",
			status: "passed",
		},
		{
			id: "check-2",
			title: "Non-Empty Prompts & Targets",
			description: "Zero null or zero-character instruction tokens",
			status: "passed",
		},
		{
			id: "check-3",
			title: "Sequence Length Boundary",
			description:
				"Tokens fit within max context window (≤ 4096 tokens)",
			status: "passed",
		},
		{
			id: "check-4",
			title: "N-Gram Overlap Leakage Check",
			description:
				"Strict deduplication between training corpus and benchmark eval splits",
			status: "passed",
		},
	];

    return (
        <div className="flex flex-col min-h-screen bg-[#f8f9fa] text-slate-800 font-sans">
            <Header
                title={`Validation Reports — ${dataset?.title || ""} @ v1.3.0`}   
                description="Inspect raw samples, manifest paths, and run gating checks"
            />
            <ScrollArea>
                <div className="flex flex-col gap-4 p-4 lg:p-6 max-w-7xl w-full mx-auto">
                    <div className="flex justify-between items-center">
                        <Button variant={"outline"} size={"sm"} className="flex items-center gap-1.5" onClick={() => window.history.back()}>
                            <ArrowLeftIcon className="w-4 h-4 text-slate-500" />
                            <span>Back to Dataset Detail</span>
                        </Button>

                        <Button variant={"outline"}>
                            <ClockClockwiseIcon className="w-3.5 h-3.5" />
                            <span>Re-Run Validation Gate</span>
                        </Button>
                    </div>

                    <div className="space-y-6">
                        {/* Banner Status Passed */}
                        <div className="bg-emerald-50/70 border border-emerald-200/80 rounded-3xl p-5 md:p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                            <div className="flex items-center gap-4">
                                <div className="p-3 bg-emerald-600 text-white rounded-2xl shadow-xs shrink-0">
                                    <ShieldCheckIcon className="w-6 h-6" />
                                </div>
                                <div className="space-y-1">
                                    <div className="flex items-center gap-2.5 flex-wrap">
                                        <h2 className="text-base font-bold text-slate-900 tracking-tight">
                                            Validation Gate: PASSED
                                        </h2>
                                        <span className="font-mono text-xs font-semibold text-slate-700 bg-white/80 border border-slate-200 px-2.5 py-0.5 rounded-md">
                                            defnex-defense-scenarios @ v1.2.0
                                        </span>
                                    </div>
                                    <p className="text-xs text-slate-600">
                                        Passed all gating checks. Dataset is ready for
                                        Supervised Fine-Tuning (SFT).
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-center gap-1.5 bg-white/80 border border-slate-200/80 px-3 py-1.5 rounded-full text-xs font-mono text-slate-600 shadow-xs shrink-0">
                                <ClockIcon className="w-3.5 h-3.5 text-slate-400" />
                                <span>9/23/2026, 4:13:09 PM</span>
                            </div>
                        </div>

                        {/* Grid Section */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Left Card: Row Status Counts */}
                            <div className="bg-white rounded-3xl border border-slate-200/80 p-6 shadow-xs flex flex-col justify-between space-y-6">
                                <div>
                                    <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">
                                        ROW STATUS COUNTS
                                    </h3>

                                    <div className="grid grid-cols-3 gap-3">
                                        <div className="bg-slate-50/80 rounded-2xl p-4 border border-slate-100">
                                            <p className="text-[11px] font-semibold text-slate-400 mb-1">
                                                Total Rows
                                            </p>
                                            <p className="text-xl font-bold text-slate-900">
                                                15, 228
                                            </p>
                                        </div>

                                        <div className="bg-emerald-50/60 rounded-2xl p-4 border border-emerald-100/80">
                                            <p className="text-[11px] font-semibold text-emerald-700 mb-1">
                                                Valid Samples
                                            </p>
                                            <p className="text-xl font-bold text-emerald-700">
                                                15, 228
                                            </p>
                                        </div>

                                        <div className="bg-rose-50/60 rounded-2xl p-4 border border-rose-100/80">
                                            <p className="text-[11px] font-semibold text-rose-700 mb-1">
                                                Invalid / Corrupt
                                            </p>
                                            <p className="text-xl font-bold text-rose-700">
                                                0
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                {/* Data Leakage Subsection */}
                                <div className="space-y-3 pt-4 border-t border-slate-100">
                                    <div className="flex justify-between items-center">
                                        <span className="text-xs font-bold text-slate-700">
                                            Train/Test Data Leakage
                                        </span>
                                        <span className="text-[10px] font-bold tracking-wider text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full border border-emerald-200/60 uppercase">
                                            ZERO LEAKAGE (CLEAN)
                                        </span>
                                    </div>

                                    <div className="p-3.5 bg-slate-50 rounded-2xl border border-slate-100 text-xs font-mono text-slate-500">
                                        0 train/eval contamination detected across strict
                                        n-gram hash matrix.
                                    </div>
                                </div>
                            </div>

                            {/* Right Card: Detailed Gate Verification Checks */}
                            <div className="bg-white rounded-3xl border border-slate-200/80 p-6 shadow-xs space-y-4">
                                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
                                    DETAILED GATE VERIFICATION CHECKS
                                </h3>

                                {/* List Items */}
                                <div className="space-y-3">
                                    {verificationChecks.map((item) => (
                                        <div
                                            key={item.id}
                                            className="p-4 rounded-2xl bg-slate-50/60 border border-slate-100 flex items-center justify-between gap-4 transition-all hover:border-slate-200">
                                            <div className="space-y-0.5">
                                                <h4 className="text-xs font-bold text-slate-800">
                                                    {item.title}
                                                </h4>
                                                <p className="text-[11px] text-slate-500">
                                                    {item.description}
                                                </p>
                                            </div>

                                            <div className="shrink-0 text-emerald-600">
                                                <CheckCircleIcon className="w-5 h-5" />
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </ScrollArea>
        </div>
    )
}