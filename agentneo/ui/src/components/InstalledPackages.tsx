import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";

const InstalledPackages = ({ packages }) => {
  if (!packages) {
    return <div>Loading installed packages...</div>;
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
                <TableHead>PACKAGE</TableHead>
                <TableHead>VERSION</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {packages.map((pkg, index) => (
                <TableRow key={index}>
                  <TableCell>{pkg.name}</TableCell>
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