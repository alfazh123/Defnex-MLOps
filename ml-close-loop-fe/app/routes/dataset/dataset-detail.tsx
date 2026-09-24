import clsx from "clsx";
import { useEffect, useState } from "react";
import { useParams } from "react-router";
import Header from "~/components/ui/header";
import { ScrollArea } from "~/components/ui/scroll-area";
import { Textarea } from "~/components/ui/textarea";
import type { Dataset } from "~/type";
import { datasets } from "~/utils";
import { Button } from "~/components/ui/button";
import { ArrowLeftIcon, CheckCircleIcon, CheckIcon, ClockClockwiseIcon, ClockIcon, CopySimpleIcon, DatabaseIcon, FileTextIcon, HeartbeatIcon, KeyIcon, PlayIcon, WarningIcon } from "@phosphor-icons/react/dist/ssr";

export default function DatasetDetail() {
    const params = useParams();
    const slug = params.id;

    const [dataset, setDataset] = useState<Dataset>();
    const [copiedKey, setCopiedKey] = useState<string | null>(null);

    useEffect(() => {
        const fetchDatasetDetail = async () => {
            const data: Dataset = datasets.find((d) => d.id === slug) as Dataset;
            setDataset(data);
        }
        fetchDatasetDetail();
    }, [slug]);

    function copyToClipboard(text: string, key: string) {
        navigator.clipboard.writeText(text);
        setCopiedKey(key);
        window.setTimeout(() => setCopiedKey(null), 2000);
    }

    return (
        <div className="flex flex-col min-h-screen bg-[#f8f9fa] p-4 lg:p-6 text-slate-800 font-sans">
            <Header
                title={`Dataset Manifest ${dataset?.title || ""} @ v1.3.0`}   
                description="Inspect raw samples, manifest paths, and run gating checks"
            />
            <ScrollArea className="flex-1 mt-4">
                <div className="max-w-7xl mx-auto space-y-6 pb-10">
                    <div className="flex justify-between">
                        <Button variant={"outline"} size={"sm"} onClick={() => window.history.back()}>
                            <ArrowLeftIcon /> Back to Datasets
                        </Button>
                        <div>
                            <Button variant={"secondary"} size={"sm"} className="ml-2" >
                                <ClockClockwiseIcon className="w-3.5 h-3.5 mr-1" /> History Validation
                            </Button>
                            <a href={`/dataset/${dataset?.id}/validation-history`} rel="noopener noreferrer" className="cursor-pointer"> 
                                <Button variant={"outline"} size={"sm"} className="ml-2" >
                                    <HeartbeatIcon className="w-3.5 h-3.5 mr-1" /> Run Validate SFT
                                </Button>
                            </a>
                            <Button variant={"default"} size={"sm"} className="ml-2" >
                                <PlayIcon className="w-3.5 h-3.5 mr-1" /> Launch SFT RUN with this version
                            </Button>
                        </div>
                    </div>
                    <div className="grid md:grid-cols-3 gap-6">
                        {/* Left Column: Metadata & Details */}
                        <div className="md:col-span-2 bg-white rounded-3xl border border-slate-200/80 p-6 md:p-8 shadow-xs flex flex-col gap-6">
                            
                            {/* Card Header: Title & Status */}
                            <div className="flex w-full justify-between items-start pb-6 border-b border-slate-100">
                                <div className="flex flex-col gap-1">
                                    <span className="text-xs font-bold text-gray-700 bg-indigo-50 px-2.5 py-1 rounded-full w-fit">
                                        v1.3.0
                                    </span>
                                    <h2 className="text-xl font-bold text-slate-900 mt-1">
                                        {dataset?.title}
                                    </h2>
                                    <p className="text-xs text-slate-400 font-mono">
                                        Dataset ID: {dataset?.id}
                                    </p>
                                </div>
                                <div className={clsx(
                                    "px-3 py-1 text-xs font-semibold rounded-full flex items-center gap-1.5 border shrink-0",
                                    dataset?.sftStatus === "Ready" ? "bg-emerald-50 text-emerald-600 border-emerald-200" :
                                    dataset?.sftStatus === "Queued" ? "bg-amber-50 text-amber-600 border-amber-200" :
                                    dataset?.sftStatus === "Failed" ? "bg-rose-50 text-rose-600 border-rose-200" :
                                    "bg-slate-50 text-slate-600 border-slate-200"
                                )}>
                                    {dataset?.sftStatus === "Ready" && <CheckCircleIcon className="w-3.5 h-3.5" />}
                                    {dataset?.sftStatus === "Queued" && <ClockIcon className="w-3.5 h-3.5" />}
                                    {dataset?.sftStatus === "Failed" && <WarningIcon className="w-3.5 h-3.5" />}
                                    <span>{dataset?.sftStatus}</span>
                                </div>
                            </div>

                            {/* Stat Highlights */}
                            <div className="grid grid-cols-2 gap-4 p-4 rounded-2xl bg-slate-50/80 border border-slate-100">
                                <div>
                                    <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-0.5">
                                        Total Samples
                                    </p>
                                    <div className="text-2xl font-bold text-slate-900">
                                        {dataset?.totalRows?.toLocaleString() || "0"}
                                    </div>
                                </div>
                                <div>
                                    <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-0.5">
                                        Manifest Format
                                    </p>
                                    <div className="text-2xl font-bold text-black uppercase">
                                        {dataset?.format}
                                    </div>
                                </div>
                            </div>

                            {/* Storage & Manifest Details */}
                            <div className="space-y-4">
                                <div className="space-y-1">
                                    <p className="text-xs font-semibold text-slate-500 flex items-center gap-1.5">
                                        <DatabaseIcon className="w-3.5 h-3.5 text-slate-400" /> Storage URL
                                    </p>
                                    <div className="relative p-3 bg-slate-50 rounded-xl text-xs font-mono text-slate-600 break-all border border-slate-100">
                                        s3://defnex-mlops/datasets/defnex-defense-scenarios/v1.2.0/
                                        <Button variant={"ghost"} className="absolute top-0 right-0 p-1 rounded-full hover:bg-slate-100" size="icon" onClick={() => copyToClipboard("s3://defnex-mlops/datasets/defnex-defense-scenarios/v1.2.0/", "storage-url") }>
                                            {copiedKey === "storage-url" ? (
                                                <CheckIcon className="w-3.5 h-3.5 text-emerald-500" />
                                            ) : (
                                                <CopySimpleIcon className="w-3.5 h-3.5 text-slate-400 hover:text-slate-600" />
                                            )}
                                        </Button>
                                    </div>
                                </div>

                                <div className="grid md:grid-cols-2 gap-4">
                                    <div className="space-y-1">
                                        <p className="text-xs font-semibold text-slate-500 flex items-center gap-1.5">
                                            <FileTextIcon className="w-3.5 h-3.5 text-slate-400" /> Manifest Path
                                        </p>
                                        <div className="relative p-3 bg-slate-50 rounded-xl text-xs font-mono text-slate-600 truncate border border-slate-100" title="s3://defnex-mlops/datasets/defnex-defense-scenarios/v1.2.0/manifest.json">
                                            s3://defnex-mlops/.../manifest.json
                                            <Button variant={"ghost"} className="absolute top-0 right-0 p-1 rounded-full hover:bg-slate-100" size="icon" onClick={() => copyToClipboard("s3://defnex-mlops/datasets/defnex-defense-scenarios/v1.2.0/manifest.json", "manifest-path") }>
                                                {copiedKey === "manifest-path" ? (
                                                    <CheckIcon className="w-3.5 h-3.5 text-emerald-500" />
                                                ) : (
                                                    <CopySimpleIcon className="w-3.5 h-3.5 text-slate-400 hover:text-slate-600" />
                                                )}
                                            </Button>
                                        </div>
                                    </div>
                                    <div className="space-y-1">
                                        <p className="text-xs font-semibold text-slate-500 flex items-center gap-1.5">
                                            <KeyIcon className="w-3.5 h-3.5 text-slate-400" /> SHA-256 Hash
                                        </p>
                                        <div className="relative p-3 bg-slate-50 rounded-xl text-xs font-mono text-slate-600 truncate border border-slate-100">
                                            6d2258a41920da78794d5a7678d3b055
                                            <Button variant={"ghost"} className="absolute top-0 right-0 p-1 rounded-full hover:bg-slate-100" size="icon" onClick={() => copyToClipboard("6d2258a41920da78794d5a7678d3b055", "sha-256-hash") }>
                                                {copiedKey === "sha-256-hash" ? (
                                                    <CheckIcon className="w-3.5 h-3.5 text-emerald-500" />
                                                ) : (
                                                    <CopySimpleIcon className="w-3.5 h-3.5 text-slate-400 hover:text-slate-600" />
                                                )}
                                            </Button>
                                        </div>
                                    </div>
                                </div>

                                <div className="space-y-1.5 pt-2">
                                    <p className="text-xs font-semibold text-slate-500">
                                        Intake Notes
                                    </p>
                                    <Textarea 
                                        readOnly 
                                        value={"Intake for DEFNEX Scenario v1 fine-tuning"} 
                                        className="bg-slate-50 border-slate-200/80 rounded-2xl text-xs text-slate-700 resize-none focus-visible:ring-0"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Right Column: Code Inspection */}
                        <div className="bg-slate-900 rounded-3xl p-6 shadow-sm flex flex-col gap-4 text-slate-100 border border-slate-800">
                            <div className="flex justify-between items-center pb-3 border-b border-slate-800">
                                <p className="text-xs font-semibold tracking-wide text-slate-400">
                                    Manifest Sample Inspection
                                </p>
                                <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded-full border border-emerald-800/50">
                                    {dataset?.format}
                                </span>
                            </div>
                            <div className="flex-1 overflow-hidden">
                                <pre className="bg-slate-950 p-4 rounded-2xl overflow-x-auto text-[11px] font-mono leading-relaxed text-emerald-300 border border-slate-800/80 max-h-[500px]">
                                    <code>
{`{
  "scenario_id": "SCN-DEF-2026-089",
  "domain": "coastal_surveillance",
  "environment": "maritime_littoral",
  "time": "2026-08-14T03:45:00Z",
  "location": "Natuna North Sector B-4",
  "objective": "Identify dark vessel incursions",
  "entities": [
    "P-8 Patrol Drone Alpha",
    "Fast Patrol Boat KRI-628"
  ],
  "indicators": [
    "Radar cross section 45m steel hull",
    "AIS disabled for 6 hours"
  ],
  "recommended_actions": [
    "Dispatch FPB KRI-628 for intercept",
    "Maintain electro-optical track"
  ],
  "version": "v1.2.0"
}`}
                                    </code>
                                </pre>
                            </div>
                        </div>
                    </div>
                </div>
            </ScrollArea>
        </div>
    );
}