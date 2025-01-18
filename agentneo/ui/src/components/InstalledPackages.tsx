import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";

interface PackageInfo {
  name: string;
  version: string;
}

interface InstalledPackagesProps {
  packages: PackageInfo[];
}

export const InstalledPackages: React.FC<InstalledPackagesProps> = ({ packages }) => {
  if (!packages || packages.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Installed Packages</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-gray-500">No packages found</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Installed Packages</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[350px]">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[200px]">Package</TableHead>
                <TableHead>Version</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {packages.map((pkg, index) => (
                <TableRow key={index}>
                  <TableCell className="font-medium">{pkg.name}</TableCell>
                  <TableCell>{pkg.version}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default InstalledPackages;