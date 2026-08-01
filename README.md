# SPSC-RNA-seq-manip

# OBJECTIVES
ATERA

- Differences between visum, xenium, and ATERA in terms of chemistry of the reaction, transcriptomic probe panel (coverage), design of custom probes, resolution of the images, number of samples that can be processed.
- Padlock probes and bits (codeword bits) differences with Xenium?
- How combine gene expression data and imaging data (applications)
- Software to do data analysis (website vs xenium software). They are only for visualization (explore how to do that and the options available) or your can include coding-based analysis using that softwares? if data analysis is required to be done separately, how to do data analysis and possible outputs of the analysis for the dataset. 
- How to study differential gene expression (software and pipelines) used. How to detect rare and transient cell states?
- Cell to cell communication and interactions.
- Spatial gene expression gradients
- How to do cell annotation.
- Cell segmentation.
- Spatial territory definition and unsupervised tissue annotation. 
- Spatial signaling pathway mapping.
- Download any kind of user manual, protocol, and any useful pdf information. Share it in a organized way (folders) to upload that to the CFCE Dropbox.
- Do a presentation and/or word document  explaining things commented above.


## NOTES
- genes are not assigned to cell id by defalt. use cell boundries paquet to group genes by cell_id
- chunking: sort by x value, copy until limit reached. sort by y value
- wtf is going on with the negative controll probes

- Raw transcripts.parquet
          |
          ↓
- Assign transcripts → cells
          |
          ↓
- Create cell × gene matrix
          |
          ↓
- AnnData
          |
          ↓
- Scanpy
(cell clustering/type)
          |
          ↓
- Squidpy (spatial neighbors)
          |
          ↓
- LIANA (ligand/receptor inference)
          |
          ↓
- Network analysis
          |
          ↓
- Visualization

goal