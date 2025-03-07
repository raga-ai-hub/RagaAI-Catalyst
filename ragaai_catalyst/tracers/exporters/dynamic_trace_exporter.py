"""
Dynamic Trace Exporter - A wrapper for RAGATraceExporter that allows dynamic updates to properties.
"""
import logging
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from ragaai_catalyst.tracers.exporters.ragaai_trace_exporter import RAGATraceExporter

logger = logging.getLogger("RagaAICatalyst")

class DynamicTraceExporter(SpanExporter):
    """
    A wrapper around RAGATraceExporter that allows dynamic updates to properties.
    This exporter forwards all calls to the underlying RAGATraceExporter but allows
    certain properties to be updated dynamically during execution.
    """
    
    def __init__(self, files_to_zip, project_name, project_id, dataset_name, user_details, base_url, custom_model_cost):
        """
        Initialize the DynamicTraceExporter.
        
        Args:
            files_to_zip: List of files to zip
            project_name: Project name
            project_id: Project ID
            dataset_name: Dataset name
            user_details: User details
            base_url: Base URL for API
        """
        self._exporter = RAGATraceExporter(
            files_to_zip=files_to_zip,
            project_name=project_name,
            project_id=project_id,
            dataset_name=dataset_name,
            user_details=user_details,
            base_url=base_url,
            custom_model_cost=custom_model_cost
        )
        
        # Store the initial values
        self._files_to_zip = files_to_zip
        self._project_name = project_name
        self._project_id = project_id
        self._dataset_name = dataset_name
        self._user_details = user_details
        self._base_url = base_url
        self._custom_model_cost = custom_model_cost

    
    def export(self, spans):
        """
        Export spans by forwarding to the underlying exporter.
        Before exporting, update the exporter's properties with the current values.
        
        Args:
            spans: Spans to export
            
        Returns:
            SpanExportResult: Result of the export operation
        """
        try:
            # Update the exporter's properties
            self._update_exporter_properties()
        except Exception as e:
            raise Exception(f"Error updating exporter properties: {e}")

        try:
            # Forward the call to the underlying exporter
            result = self._exporter.export(spans)
            return result
        except Exception as e:
            raise Exception(f"Error exporting trace: {e}")
            


    def shutdown(self):
        """
        Shutdown the exporter by forwarding to the underlying exporter.
        Before shutting down, update the exporter's properties with the current values.
        """
        try:
            # Update the exporter's properties
            self._update_exporter_properties()
        except Exception as e:
            raise Exception(f"Error updating exporter properties: {e}")

        try:
            # Forward the call to the underlying exporter
            return self._exporter.shutdown()
        except Exception as e:
            raise Exception(f"Error shutting down exporter: {e}")
    
    def _update_exporter_properties(self):
        """
        Update the underlying exporter's properties with the current values.
        """
        self._exporter.files_to_zip = self._files_to_zip
        self._exporter.project_name = self._project_name
        self._exporter.project_id = self._project_id
        self._exporter.dataset_name = self._dataset_name
        self._exporter.user_details = self._user_details
        self._exporter.base_url = self._base_url
        self._exporter.custom_model_cost = self._custom_model_cost
    
    # Getter and setter methods for dynamic properties
    
    @property
    def files_to_zip(self):
        return self._files_to_zip
    
    @files_to_zip.setter
    def files_to_zip(self, value):
        self._files_to_zip = value
    
    @property
    def project_name(self):
        return self._project_name
    
    @project_name.setter
    def project_name(self, value):
        self._project_name = value
    
    @property
    def project_id(self):
        return self._project_id
    
    @project_id.setter
    def project_id(self, value):
        self._project_id = value
    
    @property
    def dataset_name(self):
        return self._dataset_name
    
    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value
    
    @property
    def user_details(self):
        return self._user_details
    
    @user_details.setter
    def user_details(self, value):
        self._user_details = value
    
    @property
    def base_url(self):
        return self._base_url
    
    @base_url.setter
    def base_url(self, value):
        self._base_url = value

    @property
    def custom_model_cost(self):
        return self._custom_model_cost
    
    @custom_model_cost.setter
    def custom_model_cost(self, value):
        self._custom_model_cost = value
